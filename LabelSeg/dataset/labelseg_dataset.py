import numpy as np
import json

from h5py import File

from monai.transforms import (
    RandAffineD,
    RandFlipD, Rand3DElasticD, RandZoomD, RandSimulateLowResolutionD,
    RandGaussianNoise)
from torch.utils.data import Dataset

from LabelSeg.dataset.transforms import RandDownSampleFODFOrderD

# from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


class LabelSegDataset(Dataset):
    """ Dataset for FODF volumes stored in a HDF5 file.
    """

    def __init__(
        self,
        file_path: str,
        config_file: str,
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
        self.config_file = config_file
        # TODO: Make this a parameter
        self.normalize = True
        self.is_test = is_test
        self.bundles = bundles

        # Constant for transforming angles to radians
        pi_4 = np.pi / 4

        # Transform to simulate left-right flips
        self.flip = RandFlipD(
            keys=["image", "label", "mask"], prob=0.2, spatial_axis=0)

        # Transform to simulate elastic deformations.
        # TODO: Actually use this transform right now it eats up all the memory
        self.elastic = Rand3DElasticD(
            keys=["image", "label", "mask"],
            sigma_range=[90, 120],
            magnitude_range=[9, 11],
            prob=0.2)

        # Transform to simulate affine transformations, i.e rotations and
        # translations
        self.affine = RandAffineD(
            keys=["image", "label", "mask"],
            mode=['bilinear', 'nearest', 'nearest'],
            rotate_range=pi_4,
            translate_range=5, prob=0.2)
        # Transform to simulate zooming, i.e making the image bigger or smaller
        self.zoom = RandZoomD(
            keys=["image", "label", "mask"], min_zoom=0.9, max_zoom=1.5,
            mode=['area', 'nearest-exact', 'nearest-exact'],
            prob=0.2)
        # Transform to simulate low resolution by downsampling and upsampling
        self.resize = RandSimulateLowResolutionD(
            keys=["image", "label", "mask"], zoom_range=(0.5, 1),
            upsample_mode='nearest',
            downsample_mode='nearest',
            align_corners=None,  # must be none instead of False
            prob=0.2)

        # TODO: Make this a dictionary transform as well
        self.gaussian = RandGaussianNoise(
            mean=0, std=0.05, prob=0.2)

        self.fod_down = RandDownSampleFODFOrderD(prob=0.2)

        self.bundle_ids, self.train_idx, self.val_idx, self.test_idx \
            = self._compute_length(self.config_file)

        self.bundle_set = self.bundles
        print(f'Bundles are {self.bundle_set}')

    def _get_indices(self):
        """ TODO """

        return self.train_idx, self.val_idx, self.test_idx

    def _compute_length(self, config_file):

        # Load subject ids to map them to an index
        bundle_ids = []
        train_idx, val_idx, test_idx = [], [], []
        with File(self.file_path, 'r') as f:
            with open(config_file, 'r') as g:
                config = json.load(g)

                train_split = config['train']
                val_split = config['valid']
                test_split = config['test']

                total_idx = 0
                for i, subject_id in enumerate(f.keys()):
                    bundles = self.f[subject_id]['bundles'].keys()
                    for j, b in enumerate(bundles):
                        bundle_ids.append((subject_id, b))

                        if subject_id in train_split:
                            train_idx.append(total_idx)

                        elif subject_id in val_split:
                            val_idx.append(total_idx)

                        elif subject_id in test_split:
                            test_idx.append(total_idx)

                        total_idx += 1

        return bundle_ids, train_idx, val_idx, test_idx

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

    def _get_datum(self, i):
        s, b = self.bundle_ids[i]

        return s, b

    def __getitem__(self, i):
        """ Get a FODF volume from the dataset.
        TODO: Clean this up
        """
        s, b = self._get_datum(i)

        # Get the FODF volume
        fodf_data = np.asarray(
            self.f[s]['fodf']).astype(np.float32)

        bundle_labels = np.asarray(
            self.f[s]['bundles'][f'{b}']['labels'])[None, ...].astype(
                np.float32)

        bundle_mask = bundle_labels > 0
        bundle_labels[bundle_mask] += 1

        bundle_id = self.bundle_set.index(b)
        bundle_onehot = np.zeros((len(self.bundle_set)), dtype=np.float32)
        bundle_onehot[bundle_id] = 1.
        prompt_data = bundle_onehot

        if self.normalize:
            mean = np.mean(fodf_data)
            std = np.std(fodf_data)
            fodf_data = (fodf_data - mean) / std

        data_dict = {
            'image': fodf_data, 'label': bundle_labels,
            'mask': bundle_mask
        }

        if not self.is_test:
            data_dict = self.flip(data_dict)
            # data_dict = self.elastic(data_dict)
            data_dict = self.affine(data_dict)
            data_dict = self.zoom(data_dict)
            data_dict = self.resize(data_dict)

        fodf_data, bundle_labels, bundle_mask = \
            (data_dict['image'],
             data_dict['label'],
             (data_dict['mask'] > 0.5).astype(bool))

        bundle_labels *= bundle_mask

        if not self.is_test:
            fodf_data = self.gaussian(fodf_data)
            fodf_data = self.fod_down(fodf_data)

        return fodf_data, prompt_data, bundle_labels

    def __del__(self):
        """ Close the hdf5 file
        """
        if hasattr(self, 'f'):
            self.f.close()

    def __len__(self):
        length = len(self.bundle_ids)
        return length
