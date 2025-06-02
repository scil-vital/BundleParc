#!/usr/bin/env python

"""
LabelSeg: automatic tract labelling without tractography.

This method takes as input fODF maps of order 6 of descoteaux07 basis and a WM
mask and outputs 71 bundle label maps. These maps can then be used to perform
tractometry/tract profiling/radiomics. The bundle definitions follow TractSeg's
minus the whole CC.

To cite: Antoine Théberge, Zineb El Yamani, François Rheault, Maxime Descoteaux, Pierre-Marc Jodoin (2025). LabelSeg. ISMRM Workshop on 40 Years of Diffusion: Past, Present & Future Perspectives, Kyoto, Japan.  # noqa
"""

import argparse
import nibabel as nib
import numpy as np
import os
import requests
import torch

from argparse import RawTextHelpFormatter
from pathlib import Path
from torch.nn import functional as F
from tqdm import tqdm

from scipy.ndimage import gaussian_filter, label, binary_dilation

from scilpy.io.utils import (
    assert_inputs_exist, assert_outputs_exist, add_overwrite_arg)
from scilpy.image.volume_operations import resample_volume

from LabelSeg.models.utils import get_model
from LabelSeg.utils.utils import get_device

# TODO: Get bundle list from model
DEFAULT_BUNDLES = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right', 'MCP', 'MLF_left', 'MLF_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right', 'SLF_III_left', 'SLF_III_right', 'SLF_II_left', 'SLF_II_right', 'SLF_I_left', 'SLF_I_right', 'STR_left', 'STR_right', 'ST_FO_left', 'ST_FO_right', 'ST_OCC_left', 'ST_OCC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'T_OCC_left', 'T_OCC_right', 'T_PAR_left', 'T_PAR_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PREC_left', 'T_PREC_right', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left', 'T_PREM_right', 'UF_left', 'UF_right']  # noqa E501

DEFAULT_CKPT = os.path.join('checkpoints', 'labelseg.ckpt')


def to_numpy(tensor: torch.Tensor, dtype=np.float32) -> np.ndarray:
    """ Helper function to convert a torch GPU tensor
    to numpy.
    """

    return tensor.cpu().numpy().astype(dtype)


class LabelSeg():
    """

    """

    def __init__(
        self,
        dto: dict,
    ):
        """
        """
        self.checkpoint = dto['checkpoint']
        self.img_size = 128  # TODO: get image size from model
        self.fodf = dto['in_fodf']
        self.bundles = DEFAULT_BUNDLES
        self.nb_labels = dto['nb_pts']
        self.half = dto['half_precision']
        self.out_prefix = dto['out_prefix']
        self.out_folder = dto['out_folder']
        self.pft_maps = dto['pft_maps']
        self.min_blob_size = dto['min_blob_size']
        self.keep_biggest_blob = dto['keep_biggest_blob']

        self.n_coefs = int(
            (8 + 2) * (8 + 1) // 2)  # TODO: infer sh order from model

    def _get_outer_shell(self, label, next_label):
        dilated_data = binary_dilation(label, iterations=1)
        inter_data = dilated_data * next_label

        outer_shell = dilated_data - (label + inter_data)

        return outer_shell

    def compute_pft_maps(self, label_map):
        """ Compute PFT-stype maps. 'map_include' is computed as
        the outer shell of the first and last labels. 'interface' is
        the first and last labels. 'map_exclude' is everything outside
        the bundle, including the 'map_include'.

        Parameters
        ----------
        label_map: nib.Nifti1Image
            label map

        Returns
        -------
        map_include: nib.Nifti1Image

        map_exclude: nib.Nifti1Image

        interface: nib.Nifti1Image
        """

        label_data = label_map.get_fdata(dtype=np.float32)
        bin_label = (label_data > 0).astype(np.uint8)

        first, last = 1, label_data.max()
        second, second_to_last = 2, last - 1

        first_label, second_label = label_data == first, label_data == second
        last_label, second_to_last_label = label_data == last, \
            label_data == second_to_last

        outer_first = self._get_outer_shell(
            first_label.astype(np.uint8), second_label.astype(np.uint8))
        outer_last = self._get_outer_shell(
            last_label.astype(np.uint8), second_to_last_label.astype(np.uint8))

        map_include = outer_first + outer_last
        interface = first_label + last_label
        map_exclude = 1 - ((bin_label + map_include) > 0)

        include_img = nib.Nifti1Image(
            map_include.astype(np.float32), label_map.affine, label_map.header)
        exclude_img = nib.Nifti1Image(
            map_exclude.astype(np.float32), label_map.affine, label_map.header)
        interface_img = nib.Nifti1Image(
            interface.astype(np.float32), label_map.affine, label_map.header)
        seeding_img = nib.Nifti1Image(
            bin_label.astype(np.float32), label_map.affine, label_map.header)

        return include_img, exclude_img, interface_img, seeding_img

    def post_process_mask(self, mask, bundle_name,
                          min_blob_size=100, keep_biggest_blob=False):
        """ TODO
        """
        # Get the blobs in the image. Ideally, a mask only has one blob.
        # More than one either indicates a broken segmentation, or extraneous
        # voxels. TODO: handle these cases differently.
        bundle_mask = (mask > 0.5)

        # TODO: investigate blobs
        blobs, nb = label(bundle_mask)

        # No need to process, return the mask
        if nb <= 1:
            return bundle_mask.astype(np.uint8)

        # Calculate the size of each blob
        blob_sizes = np.bincount(blobs.ravel())
        new_mask = np.zeros_like(bundle_mask)

        if keep_biggest_blob:
            print("WARNING: More than one blob, keeping largest")
            biggest_blob = np.argmax(blob_sizes[1:])
            new_mask[blobs == biggest_blob + 1] = 1
            return new_mask

        # Remove blobs under a certain size (100)
        for i in range(1, len(blob_sizes[1:])):
            if blob_sizes[i] >= min_blob_size:
                new_mask[blobs == i] = 1

        return bundle_mask.astype(np.uint8)

    def post_process_labels(
        self, bundle_label, bundle_mask, nb_labels, sigma=0.5
    ):
        """ Masked filtering (normalized convolution) and label discretizing.
        Reference:
        https://stackoverflow.com/questions/59685140/python-perform-blur-only-within-a-mask-of-image  # noqa
        """

        out_type = np.uint16 if nb_labels > 1 else np.uint8

        # Masked convolution
        float_mask = bundle_mask.astype(float)
        filtered = gaussian_filter(bundle_label * float_mask, sigma=sigma)
        weights = gaussian_filter(float_mask, sigma=sigma)
        filtered /= (weights + 1e-8)
        filtered = filtered * bundle_mask
        # Label masking
        discrete_labels = bundle_label[bundle_mask.astype(bool)]

        # Label dicretizing
        discrete_labels = np.ceil(discrete_labels * nb_labels)
        bundle_label[bundle_mask.astype(bool)] = discrete_labels
        bundle_label[~bundle_mask.astype(bool)] = 0

        return bundle_label.astype(out_type)

    @torch.no_grad()
    def predict(self, model, fodf, min_blob_size, keep_biggest_blob):
        """
        """
        nb_bundles = len(self.bundles)
        fodf_data = fodf.get_fdata().transpose(
            (3, 0, 1, 2))[:self.n_coefs, ...]  # TODO: get from model

        # z-score norm
        mean = np.mean(fodf_data)
        std = np.std(fodf_data)
        fodf_data = (fodf_data - mean) / std

        # with torch.amp.autocast('cuda', enabled=False):
        # Predict the scores of the streamlines
        pbar = tqdm(range(nb_bundles))

        with torch.amp.autocast(str(get_device())):

            data = torch.tensor(
                fodf_data,
                dtype=torch.float
            ).to('cuda:0')

            prompts = torch.eye(len(self.bundles), device=get_device())

            z, encoder_features = model.labelsegnet.encode(
                data[None, ...])

            for i in pbar:
                pbar.set_description(self.bundles[i])

                y_hat = F.sigmoid(model.labelsegnet.decode(
                    z, encoder_features, prompts[None, i, ...]
                )[-1]).squeeze()

                y_hat_np = to_numpy(y_hat)
                bundle_mask = y_hat_np[0]
                bundle_label = y_hat_np[1]

                bundle_mask = self.post_process_mask(
                    bundle_mask, self.bundles[i], min_blob_size=min_blob_size,
                    keep_biggest_blob=keep_biggest_blob)
                bundle_label = self.post_process_labels(
                    bundle_label, bundle_mask, self.nb_labels)

                yield bundle_label, self.bundles[i]

    def run(self):
        """
        Main method where the magic happens
        """
        fodf_in = nib.load(self.fodf)

        # TODO in future release: infer sh order from model
        n_coefs = 45
        X, Y, Z, C = fodf_in.get_fdata().shape

        assert C >= n_coefs, \
            f'Input fODFs should have at least {n_coefs} coefficients.'

        if C > n_coefs:
            print('Input fODFs have more than 28 coefficients. '
                  f'Only the first {n_coefs} will be used.')

        # Resampling volume to fit the model's input at training time
        resampled_img = resample_volume(fodf_in, ref_img=None,
                                        volume_shape=[self.img_size],
                                        iso_min=False,
                                        voxel_res=None,
                                        interp='lin',
                                        enforce_dimensions=False)

        # Load the model
        model = get_model(self.checkpoint, {'pretrained': True})

        # Predict label maps. `self.predict` is a generator
        # yielding one label map per bundle (and a binary mask)
        for y_hat_label, b_name in self.predict(
            model, resampled_img, self.min_blob_size, self.keep_biggest_blob
        ):

            Path(os.path.join(self.out_folder, b_name)).mkdir(exist_ok=True)

            # Format the output as a nifti image
            label_img = nib.Nifti1Image(y_hat_label,
                                        resampled_img.affine,
                                        resampled_img.header, dtype=np.uint16)

            # Resampling volume to fit the model's input at training time
            resampled_label = resample_volume(label_img, ref_img=None,
                                              volume_shape=[X, Y, Z],
                                              iso_min=False,
                                              voxel_res=None,
                                              interp='nn',
                                              enforce_dimensions=False)
            # Save it
            nib.save(resampled_label, os.path.join(
                self.out_folder, b_name, 'labels_map.nii.gz'))

            if self.pft_maps:
                map_include, map_exclude, interface, seeding_img = \
                        self.compute_pft_maps(resampled_label)

                nib.save(map_include, os.path.join(
                    self.out_folder, b_name, 'map_include.nii.gz'))

                nib.save(map_exclude, os.path.join(
                    self.out_folder, b_name, 'map_exclude.nii.gz'))

                nib.save(interface, os.path.join(
                    self.out_folder, b_name, 'interface.nii.gz'))

                nib.save(seeding_img, os.path.join(
                    self.out_folder, b_name, 'seeding.nii.gz'))


def _build_arg_parser(parser):
    parser.add_argument('in_fodf', type=str,
                        help='fODF input of order 6 or above. If the input '
                             'has more than 28 coefficients, only the first 28'
                             ' will be used.')
    parser.add_argument('--out_prefix', type=str, default='',
                        help='Output file prefix. Default is nothing. ')
    parser.add_argument('--out_folder', type=str, default='.',
                        help='Output destination. Default is the current '
                             'directory.')
    parser.add_argument('--pft_maps', action='store_true',
                        help='Output PFT-stype maps to track in predicted '
                             'bundles.')
    parser.add_argument('--nb_pts', type=int, default=50,
                        help='Number of divisions per bundle. '
                             'Default is [%(default)s].')
    parser.add_argument('--half_precision', action='store_true',
                        help='Use half precision (float16) for inference. '
                             'This reduces memory usage but may lead to '
                             'reduced accuracy.')
    parser.add_argument('--min_blob_size', type=int, default=100,
                        help='Mininum blob size to keep.')
    parser.add_argument('--keep_biggest_blob', action='store_true',
                        help='Only keep the biggest blob.')
    parser.add_argument('--checkpoint', type=str,
                        default=DEFAULT_CKPT,
                        help='Checkpoint (.ckpt) containing hyperparameters '
                             'and weights of model. Default is '
                             '[%(default)s]. If the file does not exist, it '
                             'will be downloaded.')
    add_overwrite_arg(parser)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter)

    _build_arg_parser(parser)
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_fodf)
    assert_outputs_exist(parser, args, [], [args.out_prefix])

    return parser, args


def _download_weights(path=DEFAULT_CKPT):
    url = 'https://zenodo.org/records/14779787/files/labelseg.ckpt'
    os.makedirs(os.path.dirname(path))
    print('Downloading weights ...')
    with requests.get(url, stream=True) as r:
        with open(path, 'wb') as f:
            f.write(r.content)
    print('Done !')


def main():

    parser, args = parse_args()

    # if file not exists
    if not os.path.exists(args.checkpoint):
        _download_weights(args.checkpoint)

    experiment = LabelSeg(vars(args))
    experiment.run()


if __name__ == "__main__":
    main()
