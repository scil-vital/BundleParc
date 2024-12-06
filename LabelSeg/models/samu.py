# import nibabel as nib
# import numpy as np
import torch

import torch.nn.functional as F
from typing import List

from lightning.pytorch import LightningModule
from torch.nn import BCELoss, MSELoss

from monai.losses import DiceLoss
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU
from timm.scheduler.cosine_lr import CosineLRScheduler

from LabelSeg.models.labelsegnet import LabelSegNet


class LabelSeg(LightningModule):

    def __init__(
        self,
        prompt_strategy: str,
        volume_size: int = 96,
        bundles: list = [],
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.1,
        warmup_t: int = 5,
        epochs: int = 1000,
        model=None,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Parameters:
        -----------
        """
        super(LabelSeg, self).__init__()
        # Keep the name of the model
        self.hparams["name"] = self.__class__.__name__
        self.pretrained = False
        self.prompt_strategy = prompt_strategy
        self.volume_size = volume_size
        self.bundles = bundles

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_t = warmup_t
        self.epochs = epochs

        self.ce = BCELoss(reduction='none')
        self.mse = MSELoss(reduction='none')
        self.dice = DiceLoss(sigmoid=False, jaccard=False, squared_pred=True,
                             reduction='none', include_background=True)

        self.dice_metric = DiceMetric(
            include_background=True, reduction='mean_channel',
            ignore_empty=False, num_classes=2)
        self.iou_metric = MeanIoU(ignore_empty=False)

        self.labelsegnet = model
        if model is None:
            self.labelsegnet = LabelSegNet(28, volume_size, prompt_strategy,
                                    n_bundles=len(self.bundles))

        if not self.pretrained:
            self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.labelsegnet.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay,
                                      betas=(0.9, 0.999))
        scheduler = CosineLRScheduler(optimizer, t_initial=5, lr_min=1e-7,
                                      cycle_limit=self.epochs,
                                      warmup_t=self.warmup_t,
                                      warmup_lr_init=1e-7)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
        # timm's scheduler need the epoch value

    def loss(self, y_pred, y_true):
        """ Compute the loss for the model. The loss is a combination of
        Cross Entropy and Dice Loss. Dice Loss is computed only on the
        first channel of the output, cross entropy on the
        second. The CE is computed only on the labels, not the background.

        The targets are computed from the label map, which
        has values from 1 to 2 for labels and 0 for background.

        Parameters:
        -----------
        y_pred: torch.Tensor
            The output of the model
        y_true: torch.Tensor
            The target of the model

        Returns:
        --------
        mse: torch.Tensor
            The mean squared error of the model
        dice: torch.Tensor
            The dice loss of the model
        """

        # Mask prediction
        y_mask = y_pred[:, [0]]
        # Mask for the MSE
        mse_mask = (y_true >= 1)
        # Target for mask
        y = mse_mask.to(int)

        # Labels prediction
        y_labels = y_pred[:, [1]]
        # Target for labels. The target is the label - 1
        # to project the labels from 1-2 to 0-1. The
        # background is not considered in the loss and is
        # clipped to 0.
        y_ce = torch.clip(y_true[mse_mask] - 1, 0, 1)
        y_hat_ce = y_labels[mse_mask]

        # Compute the loss
        mse = self.ce(y_hat_ce, y_ce).mean()
        dice = self.dice(y_mask, y).mean()
        return mse, dice

    def mask_loss(self, y_pred, y_true):
        # First loss at full scale
        mse, dice = self.loss(y_pred[-1], y_true)
        # Size of "F"ull res
        _, _, FH, FD, FW = y_true.size()
        # Compute loss at every scale by interpolating the target
        for y_hat in y_pred[-2::-1]:
            # Size of smaller scale
            _, _, H, D, W = y_hat.size()
            # Factor to scale the loss
            # TODO: Scale loss by factor cubed or squared instead of linearly ?
            # To account for the number of voxels ?
            factor = H / FH
            # Interpolate target to the size of the current scale
            y = F.interpolate(y_true, (H, W, D), mode='nearest')
            # Compute loss and add it to the total loss
            l_mse, l_dice = self.loss(y_hat, y)
            mse += factor * l_mse
            dice += factor * l_dice
        # Return the total loss
        return mse, dice

    def forward(self, x, p, m) -> List[torch.Tensor]:
        """
        Predict labels and masks from an image and prompts.

        Parameters:
        -----------
        x: torch.Tensor
            The input image
        p: torch.Tensor
            The input prompts
        m: torch.Tensor
            The input mask

        Returns:
        --------
        y: List[torch.Tensor]
            The predicted labels and masks at different scales
        """
        y = self.labelsegnet(x, p, m)

        return y

    def training_step(self, train_batch, batch_idx):
        """ Compute the loss for the model and log it. """
        x_i, x_p, w_m, y = train_batch

        y_hat = self.forward(x_i, x_p, w_m)
        loss_mse, loss_dice = self.mask_loss(y_hat, y)

        loss = loss_mse + loss_dice

        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('train_loss_mse', loss_mse, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('train_loss_dice', loss_dice, on_step=False, on_epoch=True,
                 sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x_i, x_p, w_m, y = val_batch

        y_hat = self.forward(x_i, x_p, w_m)
        loss_mse, loss_dice = self.mask_loss(y_hat, y)
        loss = loss_mse + loss_dice

        preds = (y_hat[-1][:, [0]] > 0.5).int()
        y_mask = (y >= 1).int()

        mean_dice = self.dice_metric(preds, y_mask).mean()
        mean_iou = self.iou_metric(preds, y_mask).mean()

        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('val_loss_dice', loss_dice, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('val_loss_mse', loss_mse, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('val_dice', mean_dice, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('val_iou', mean_iou, on_epoch=True, on_step=False,
                 sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        x_i, x_p, w_m, y = test_batch

        y_hat = self.forward(x_i, x_p, w_m)
        loss_mse, loss_dice = self.mask_loss(y_hat, y)
        loss = loss_mse + loss_dice

        preds = (y_hat[-1][:, [0]] > 0.5).int()
        y_mask = (y >= 1).int()

        mean_dice = self.dice_metric(preds, y_mask).mean()
        mean_iou = self.iou_metric(preds, y_mask).mean()

        self.log('test_loss', loss, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('test_loss_dice', loss_dice, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('test_loss_mse', loss_mse, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('test_dice', mean_dice, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('test_iou', mean_iou, on_epoch=True, on_step=False,
                 sync_dist=True)
