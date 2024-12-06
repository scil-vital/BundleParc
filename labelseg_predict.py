#!/usr/bin/env python
import argparse
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch

from argparse import RawTextHelpFormatter
from tqdm import tqdm

from scipy.ndimage import gaussian_filter, binary_closing

from scilpy.io.utils import (
    assert_inputs_exist, assert_outputs_exist, add_overwrite_arg)
from scilpy.image.volume_operations import resample_volume

from LabelSeg.models.utils import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cast_device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        self.img_size = dto['img_size']
        self.fodf = dto['fodf']
        self.wm = dto['wm']
        self.out = dto['out']
        self.bundles = dto['bundles']
        self.nb_labels = dto['nb_labels']
        self.n_coefs = int(
            (dto['sh_order'] + 2) * (dto['sh_order'] + 1) // 2)

    def post_process_mask(self, mask, bundle_name):
        """ TODO
        """
        # Get the blobs in the image. Ideally, a mask only has one blob.
        # More than one either indicates a broken segmentation, or extraneous
        # voxels. TODO: handle these cases differently.
        bundle_mask = (mask > 0.5)

        # # TODO: investigate blobs
        # blobs, nb = ndi.label(bundle_mask)

        # # No need to process, return the mask
        # if nb <= 1:
        #     return bundle_mask.astype(np.uint8)

        # # Calculate the size of each blob
        # blob_sizes = np.bincount(blobs.ravel())
        # new_mask = np.zeros_like(bundle_mask)
        # # Remove blobs under a certain size (100)
        # for i in range(1, len(blob_sizes[1:])):
        #     if blob_sizes[i] > 1:
        #         new_mask[blobs == i] = 1

        return bundle_mask.astype(np.uint8)

    def post_process_labels(
        self, bundle_label, bundle_mask, nb_labels, sigma=0.5
    ):
        """ TODO: smooth label
        """

        float_mask = bundle_mask.astype(float)
        # Comment
        filtered = gaussian_filter(bundle_label * float_mask, sigma=sigma)

        # Comment
        weights = gaussian_filter(float_mask, sigma=sigma)

        # Comment
        filtered /= (weights + 1e-8)

        # Comment
        filtered = filtered * bundle_mask

        discrete_labels = bundle_label[bundle_mask.astype(bool)]

        discrete_labels = np.round(discrete_labels * nb_labels)

        bundle_label[bundle_mask.astype(bool)] = discrete_labels
        bundle_label[~bundle_mask.astype(bool)] = 0

        return bundle_label.astype(np.uint16)

    @torch.no_grad()
    def predict(self, model, fodf, wm):
        """
        """
        nb_bundles = len(self.bundles)
        fodf_data = fodf.get_fdata().transpose(
            (3, 0, 1, 2))[:self.n_coefs, ...]  # TODO: get from model
        wm_data = wm.get_fdata()[None, ...]

        # z-score norm
        mean = np.mean(fodf_data)
        std = np.std(fodf_data)
        fodf_data = (fodf_data - mean) / std

        # with torch.amp.autocast('cuda', enabled=False):
        # Predict the scores of the streamlines
        pbar = tqdm(range(nb_bundles))

        # TODO: Load data only once, loop over all bundles and yield result
        data = torch.tensor(
            fodf_data,
            dtype=torch.float
        ).to('cuda:0')

        wm_prompt = torch.tensor(
            wm_data,
            dtype=torch.float
        ).to('cuda:0')

        with torch.no_grad():
            for i in pbar:
                pbar.set_description(self.bundles[i])

                prompt = torch.zeros(nb_bundles, device='cuda:0')
                prompt[i] = 1

                y_hat = model(
                    data[None, ...], prompt[None, ...], wm_prompt[None, ...])[-1]
                bundle_mask = y_hat[0][0].cpu().numpy().astype(np.float32)
                bundle_label = y_hat[0][1].cpu().numpy().astype(np.float32)

                bundle_mask = self.post_process_mask(bundle_mask, self.bundles[i])
                bundle_label = self.post_process_labels(
                    bundle_label, bundle_mask, self.nb_labels)

                yield bundle_mask, bundle_label, self.bundles[i]

    def run(self):
        """
        Main method where the magic happens
        """
        fodf_in = nib.load(self.fodf)
        wm_in = nib.load(self.wm)

        # shape = fodf_in.get_fdata().shape[:3]

        # Resampling volume
        resampled_img = resample_volume(fodf_in, ref_img=None,
                                        volume_shape=[self.img_size],
                                        iso_min=False,
                                        voxel_res=None,
                                        interp='lin',
                                        enforce_dimensions=False)

        # Resampling volume
        resampled_wm = resample_volume(wm_in, ref_img=None,
                                       volume_shape=[self.img_size],
                                       iso_min=False,
                                       voxel_res=None,
                                       interp='nn',
                                       enforce_dimensions=False)

        model = get_model(self.checkpoint, {'pretrained': True})

        for y_hat_mask, y_hat_label, b_name in self.predict(
            model, resampled_img, resampled_wm
        ):

            mask_img = nib.Nifti1Image(y_hat_mask,
                                       resampled_wm.affine,
                                       resampled_wm.header)

            label_img = nib.Nifti1Image(y_hat_label,
                                        resampled_wm.affine,
                                        resampled_wm.header)

            mask_img = resample_volume(mask_img, ref_img=wm_in,
                                       # volume_shape=shape,
                                       iso_min=False,
                                       voxel_res=None,
                                       interp='nn',
                                       enforce_dimensions=False)

            label_img = resample_volume(label_img, ref_img=wm_in,
                                        # volume_shape=shape,
                                        iso_min=False,
                                        voxel_res=None,
                                        interp='nn',
                                        enforce_dimensions=False)

            nib.save(mask_img, self.out + f'_{b_name}.nii.gz')
            nib.save(
                label_img,
                self.out + f'_{b_name}_labels.nii.gz')


def _build_arg_parser(parser):
    parser.add_argument('fodf', type=str,
                        help='fODF input')
    parser.add_argument('wm', type=str,
                        help='WM input')
    parser.add_argument('out', type=str,
                        help='Output file.')
    parser.add_argument('--nb_labels', type=int, default=50)
    parser.add_argument('--sh_order', type=int, default=6,
                        choices=[2, 4, 6, 8],
                        help='SH order to use.')
    parser.add_argument('--img_size', type=int, default=96)
    parser.add_argument('--checkpoint', type=str,
                        default='model/tractoracle.ckpt',
                        help='Checkpoint (.ckpt) containing hyperparameters '
                             'and weights of model. Default is '
                             '[%(default)s].')
    parser.add_argument('--bundles', type=str, nargs='+',
                        default=['AF_L', 'AF_R', 'CC_Fr_1', 'CC_Fr_2', 'CC_Oc',
                                 'CC_Pa', 'CC_Pr_Po', 'CG_L', 'CG_R', 'FAT_L',
                                 'FAT_R', 'FPT_L', 'FPT_R', 'FX_L', 'FX_R',
                                 'IFOF_L', 'IFOF_R', 'ILF_L', 'ILF_R', 'MCP',
                                 'MdLF_L', 'MdLF_R', 'OR_ML_L', 'OR_ML_R',
                                 'POPT_L', 'POPT_R', 'PYT_L', 'PYT_R', 'SLF_L',
                                 'SLF_R', 'UF_L', 'UF_R'],
                        help='Bundle list to predict.')

    add_overwrite_arg(parser)


def parse_args():
    """ Filter a tractogram. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    _build_arg_parser(parser)
    args = parser.parse_args()

    assert_inputs_exist(parser, args.fodf)
    assert_outputs_exist(parser, args, args.out)

    return parser, args


def main():

    parser, args = parse_args()

    experiment = LabelSeg(vars(args))
    experiment.run()


if __name__ == "__main__":
    main()
