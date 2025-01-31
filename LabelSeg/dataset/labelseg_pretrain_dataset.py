import numpy as np

from h5py import File

from monai.transforms import (
    RandAffineD,
    RandFlipD, Rand3DElasticD, RandZoomD, RandSimulateLowResolutionD,
    RandGaussianNoise)
from torch.utils.data import Dataset


class LabelSegPretrainDataset(Dataset):
    """ Dataset for FODF volumes stored in a HDF5 file.
    """

    def __init__(
        self,
        file_path: str,
        wm_drop_ratio: float = 0.5,
        bundles: list = [],
        is_test: bool = False
    ):
        """
        Parameters:
        -----------
        file_path: str
            Path to the hdf5 file containing the streamlines
        """
        self.file_path = file_path
        # TODO: Make this a parameter
        self.normalize = True
        self.is_test = is_test
        self.bundles = bundles
        self.wm_drop_ratio = wm_drop_ratio

        # Constant for transforming angles to radians
        pi_4 = np.pi / 4

        # Transform to simulate left-right flips
        self.flip = RandFlipD(
            keys=["image", "label", "wm", "mask"], prob=0.2, spatial_axis=0)
        # Transform to simulate elastic deformations.
        # TODO: Actually use this transform right now it eats up all the memory
        self.elastic = Rand3DElasticD(
            keys=["image", "label", "wm", "mask"],
            sigma_range=[90, 120],
            magnitude_range=[9, 11],
            prob=0.2)
        # Transform to simulate affine transformations, i.e rotations and
        # translations
        self.affine = RandAffineD(
            keys=["image", "label", "wm", "mask"],
            mode=['bilinear', 'nearest', 'nearest', 'nearest'],
            rotate_range=pi_4,
            translate_range=5, prob=0.2)
        # Transform to simulate zooming, i.e making the image bigger or smaller
        self.zoom = RandZoomD(
            keys=["image", "label", "wm", "mask"], min_zoom=0.9, max_zoom=1.5,
            mode=['area', 'nearest-exact', 'nearest-exact', 'nearest-exact'],
            prob=0.2)
        # Transform to simulate low resolution by downsampling and upsampling
        self.resize = RandSimulateLowResolutionD(
            keys=["image", "label", "wm", "mask"], zoom_range=(0.5, 1),
            upsample_mode='nearest',
            downsample_mode='nearest',
            align_corners=None,  # must be none instead of False
            prob=0.2)

        # TODO: Make this a dictionary transform as well
        self.gaussian = RandGaussianNoise(
            mean=0, std=0.05, prob=0.2)

        # Load subject ids to map them to an index
        self.subject_id = []
        with File(self.file_path, 'r') as f:
            for subject_id in f.keys():
                self.subject_id.append((subject_id))

        self.bundle_set = self.bundles
        print(f'Bundles are {self.bundle_set}')

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

    @ property
    def f(self):
        """ Open the hdf5 file
        """
        if not hasattr(self, 'archive'):
            self.archive = File(self.file_path, 'r')
        return self.archive

    def __getitem__(self, i):
        """ Get a FODF volume from the dataset.
        TODO: Clean this up
        """
        s = self.subject_id[i]
        b_i = np.random.choice(len(self.bundle_set))
        b = self.bundle_set[b_i]

        # Get the FODF volume
        fodf_data = np.asarray(
            self.f[s]['fodf']).astype(np.float32)

        wm_data = np.asarray(
            self.f[s]['mask_wm']).astype(np.float32)

        bundle_labels = np.asarray(
            self.f[s]['bundles'][f'{b}']['labels'])[None, ...].astype(
                np.float32)

        bundle_mask = bundle_labels > 0
        bundle_labels[bundle_mask] += 1

        bundle_id = self.bundle_set.index(b)
        bundle_onehot = np.zeros((len(self.bundle_set)))
        bundle_onehot[bundle_id] = 1.
        prompt_data = bundle_onehot

        if self.normalize:
            mean = np.mean(fodf_data)
            std = np.std(fodf_data)
            fodf_data = (fodf_data - mean) / std

        rand_wm_drop = np.random.random() < self.wm_drop_ratio

        if rand_wm_drop is True:
            wm_data = np.zeros_like(wm_data)

        data_dict = {
            'image': fodf_data, 'label': bundle_labels, 'wm': wm_data,
            'mask': bundle_mask
        }

        if not self.is_test:
            data_dict = self.flip(data_dict)
            # data_dict = self.elastic(data_dict)
            data_dict = self.affine(data_dict)
            data_dict = self.zoom(data_dict)
            data_dict = self.resize(data_dict)

        fodf_data, bundle_labels, bundle_mask, wm_data = \
            (data_dict['image'],
             data_dict['label'],
             (data_dict['mask'] > 0.5).astype(bool),
             (data_dict['wm'] > 0.5).astype(np.float32))

        bundle_labels *= bundle_mask

        if not self.is_test:
            fodf_data = self.gaussian(fodf_data)

        return fodf_data, prompt_data, wm_data, bundle_labels

    def __del__(self):
        """ Close the hdf5 file
        """
        if hasattr(self, 'f'):
            self.f.close()

    def __len__(self):
        length = len(self.subject_id)
        return length
