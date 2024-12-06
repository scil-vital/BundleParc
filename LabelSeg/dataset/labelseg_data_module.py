import lightning.pytorch as pl

from torch.utils.data import DataLoader, random_split

from LabelSeg.dataset.labelseg_dataset import LabelSegDataset


class LabelSegDataModule(pl.LightningDataModule):
    """
    """

    def __init__(
        self,
        train_file: str,
        val_file: str = None,
        test_file: str = None,
        wm_drop_ratio: float = 0.5,
        bundles=[],
        batch_size: int = 1,
        num_workers: int = 10,
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
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.wm_drop_ratio = wm_drop_ratio
        self.bundles = bundles

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
        print(self.val_file, self.test_file)
        if self.val_file is None and self.test_file is None:
            whole_data = LabelSegDataset(
                self.train_file[0], self.wm_drop_ratio, self.bundles)
            self.train, self.val, self.test = \
                random_split(whole_data, [0.70, 0.20, 0.10],)
            self.test.is_test = True
        else:
            # Assign train/val datasets for use in dataloaders
            self.train = LabelSegDataset(
                self.train_file, self.wm_drop_ratio, self.bundles)

            self.val = LabelSegDataset(
                self.val_file, self.wm_drop_ratio, self.bundles)

            # Assign test dataset for use in dataloader(s)
            self.test = LabelSegDataset(
                self.test_file, self.wm_drop_ratio, self.bundles, is_test=True)

        self.bundles = self.train.bundle_set
        assert self.train.bundle_set == self.val.bundle_set

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
