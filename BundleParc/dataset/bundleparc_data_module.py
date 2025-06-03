import lightning.pytorch as pl

from torch.utils.data import DataLoader, Subset

from BundleParc.dataset.bundleparc_dataset import BundleParcDataset
from BundleParc.dataset.bundleparc_pretrain_dataset import BundleParcPretrainDataset


class BundleParcDataModule(pl.LightningDataModule):
    """
    """

    def __init__(
        self,
        file: str,
        config_file: str,
        bundles=[],
        batch_size: int = 1,
        num_workers: int = 10,
        pretrain: bool = False
    ):
        """

        Parameters:
        -----------
        train_file: str
            Path to the hdf5 file containing the training set
        val_file: str
            Path to the hdf5 file containing the validation set
        test_file: str
            Path to the hdf5 file containing the test set
        batch_size: int, optional
            Size of the batches to use for the dataloaders
        num_workers: int, optional
            Number of workers to use for the dataloaders
        """

        super().__init__()
        self.file = file
        self.config_file = config_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.bundles = bundles
        self.pretrain = pretrain

        self.save_hyperparameters()

        self.data_loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'prefetch_factor': None,
            'persistent_workers': False,
            'pin_memory': False
        }

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        cls = BundleParcPretrainDataset if self.pretrain else BundleParcDataset
        whole_data = cls(
            self.file, self.config_file, self.bundles)

        train_idx, val_idx, test_idx = \
            whole_data.train_idx, whole_data.val_idx, whole_data.test_idx

        self.train = Subset(whole_data, train_idx)
        self.val = Subset(whole_data, val_idx)
        self.test = Subset(whole_data, test_idx)

    def train_dataloader(self):
        """ Create the dataloader for the training set
        """
        train_kwargs = self.data_loader_kwargs.copy()
        train_kwargs.update({'shuffle': True})
        return DataLoader(
            self.train,
            **train_kwargs)

    def val_dataloader(self):
        """ Create the dataloader for the validation set
        """
        return DataLoader(
            self.val,
            **self.data_loader_kwargs)

    def test_dataloader(self):
        """ Create the dataloader for the test set
        """
        return DataLoader(
            self.test,
            **self.data_loader_kwargs)

    def predict_dataloader(self):
        pass
