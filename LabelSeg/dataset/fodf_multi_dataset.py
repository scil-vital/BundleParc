import numpy as np

from h5py import File
from typing import Callable

from torch.utils.data import Dataset

from monai.transforms import RandFlip, RandSpatialCrop, Resize


class FODFDataset(Dataset):
    """ Dataset for FODF volumes stored in a HDF5 file.
    """

    def __init__(
        self,
        file_path: str,
        transforms: Callable = None,
    ):
        """
        Parameters:
        -----------
        file_path: str
            Path to the hdf5 file containing the streamlines
        """
        self.file_path = file_path
        self.transforms = transforms
        self.normalize = True

        self.flip = RandFlip(prob=0.5, spatial_axis=0)
        self.crop = RandSpatialCrop((80, 80, 80), random_center=False,
                                    random_size=True)
        self.resize = Resize((96, 96, 96), mode='trilinear')

        # Load subject ids to map them to an index
        self.subject_ids = []
        with File(self.file_path, 'r') as f:
            for subject_id in f.keys():
                self.subject_ids.append(subject_id)

    def _compute_input_size(self):
        """ Compute the size of the input data
        """
        img = self._get_one_input()
        return img.shape

    def _get_one_input(self):
        """ Get one input from the dataset.
        """

        img = self[0]
        return img

    @property
    def f(self):
        """ Open the hdf5 file
        """
        if not hasattr(self, 'archive'):
            self.archive = File(self.file_path, 'r')
        return self.archive

    def __getitem__(self, i):
        """ Get a FODF volume from the dataset.
        """
        subject_id = self.subject_ids[i]

        # Get the FODF volume
        fodf_data = np.asarray(
            self.f[subject_id]['fodf']).astype(np.float32)

        if self.normalize:
            mean = np.mean(fodf_data)
            std = np.std(fodf_data)
            fodf_data = (fodf_data - mean) / std

        fodf_data = self.zoom(fodf_data)
        fodf_data = self.crop(fodf_data)
        fodf_data = self.resize(fodf_data)

        return fodf_data

    def __del__(self):
        """ Close the hdf5 file
        """
        if hasattr(self, 'f'):
            self.f.close()

    def __len__(self):
        length = len(self.subject_ids)
        return length
