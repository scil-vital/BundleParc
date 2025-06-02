import argparse

import comet_ml  # noqa
import torch

from os.path import join

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CometLogger
# from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.trainer import Trainer

from LabelSeg.dataset.labelseg_data_module import LabelSegDataModule
from LabelSeg.models.labelseg import LabelSeg
# from LabelSeg.models.labelsegnet import TwoWayAttentionBlock3D
from LabelSeg.models.utils import get_model

# Set the default precision to float32 to
# speed up training and reduce memory usage
torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy('file_descriptor')
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

DEFAULT_BUNDLES = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right', 'MCP', 'MLF_left', 'MLF_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right', 'SLF_III_left', 'SLF_III_right', 'SLF_II_left', 'SLF_II_right', 'SLF_I_left', 'SLF_I_right', 'STR_left', 'STR_right', 'ST_FO_left', 'ST_FO_right', 'ST_OCC_left', 'ST_OCC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'T_OCC_left', 'T_OCC_right', 'T_PAR_left', 'T_PAR_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PREC_left', 'T_PREC_right', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left', 'T_PREM_right', 'UF_left', 'UF_right']  # noqa E501


def _build_arg_parser():
    # TODO: Add groups
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('path', type=str, help='Path to experiment')
    parser.add_argument('experiment', help='Name of experiment.')
    parser.add_argument('id', type=str, help='ID of experiment.')
    parser.add_argument('data', type=str, help='Path to the data.')
    parser.add_argument('config', type=str, help='Dataset configuration file.')

    # TODO: reorganize arguments into meaningful groups. Fix descriptions
    # as well.
    model_g = parser.add_argument_group('Model')
    model_g.add_argument('--in_channels', default=45, type=int,
                         help='Input channel nb.')
    model_g.add_argument('--volume_size', default=128, type=int,
                         help='Input volume size')
    model_g.add_argument('--bundles', default=DEFAULT_BUNDLES, type=str,
                         nargs='+', help='Bundles')
    model_g.add_argument('--pretrain', action='store_true',
                         help='Pretraining')
    model_g.add_argument('--channels', nargs=5, type=int,
                         default=[32, 64, 128, 256, 512],
                         help='Layer channels')
    model_g.add_argument('--checkpoint', type=str,
                         help='Weights to initialize training.')

    learn_g = parser.add_argument_group('Learning')
    learn_g.add_argument('--test', action='store_true',
                         help='Do not train, only test.')
    learn_g.add_argument('--epochs', type=int, default=100,
                         help='Number of epochs')
    learn_g.add_argument('--lr', type=float, default=0.0001,
                         help='Learning rate')
    learn_g.add_argument('--beta1', type=float, default=0.9,
                         help='Adam(W) beta1 parameter.')
    learn_g.add_argument('--beta2', type=float, default=0.999,
                         help='Adam(W) beta1 parameter.')
    learn_g.add_argument('--weight_decay', type=float, default=0.1,
                         help='Weight decay.')
    learn_g.add_argument('--layer_decay', type=float, default=0.8,
                         help='Layer decay for finetuning.')
    learn_g.add_argument('--warmup_t', type=float, default=5,
                         help='Warmup epochs.')
    learn_g.add_argument('--frequency_t', type=float, default=10,
                         help='Cosine LR frequency.')

    device_g = parser.add_argument_group('Device')
    device_g.add_argument('--batch-size', type=int, default=1,
                          help='Batch size')
    device_g.add_argument('--nodes', type=int, default=1,
                          help='Nb. of nodes.')
    device_g.add_argument('--devices', type=int, default=1,
                          help='Nb. of GPUs per nodes.')
    device_g.add_argument('--num_workers', type=int, default=10,
                          help='Num. workers.')

    return parser


def train(args, root_dir):
    """ Train the model. """

    if args.test and not args.checkpoint:
        raise ValueError('No point in testing if the model is not trained.')

    dm = LabelSegDataModule(
        args.data,
        args.config,
        bundles=args.bundles,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pretrain=args.pretrain)

    # Training
    comet_logger = CometLogger(
        project_name=args.path,
        experiment_name='-'.join((args.experiment, args.id)))

    # # Log parameters
    comet_logger.log_hyperparams({
        "model": LabelSeg.__name__,
        "epochs": args.epochs})

    # Log the learning rate during training as it will vary
    # from Cosine Annealing
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=1e-7, patience=1000,
                                        verbose=True, mode="min",
                                        strict=True,
                                        check_finite=True)

    # Define the trainer
    # Mixed precision is used to speed up training and
    # reduce memory usage

    trainer = Trainer(
        devices=args.devices, num_nodes=args.nodes, accelerator='auto',
        strategy='ddp',
        logger=comet_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=10,
        num_sanity_val_steps=5,
        max_epochs=args.epochs,
        enable_checkpointing=True,
        default_root_dir=root_dir,
        precision='bf16-mixed',
        callbacks=[lr_monitor, early_stop_callback])

    if args.checkpoint:
        model = get_model(args.checkpoint, {'pretrained': True})
        model.train()
    else:
        model = LabelSeg(
            in_chans=args.in_channels,
            volume_size=args.volume_size,
            channels=args.channels,
            bundles=args.bundles,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            warmup_t=args.warmup_t)

    if not args.test:

        # # Train the model
        trainer.fit(model, dm)
    # Test the model
    trainer.test(model, dm)


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    root_dir = join(args.path, args.experiment, args.id)
    train(args, root_dir)


if __name__ == '__main__':
    main()
