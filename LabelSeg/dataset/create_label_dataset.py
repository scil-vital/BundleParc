#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" The expected directory structure is ...
TODO
"""

import argparse

import h5py
import nibabel as nib
import numpy as np

from os.path import exists, join, split

from tqdm import tqdm

from scilpy.image.volume_operations import resample_volume, apply_transform

from LabelSeg.utils.constants import TRACTSEG_BUNDLES


def create_dataset(
    subs, output_file, volume_size, nb_coeffs, dtype, bundles
):
    """
    """

    out_dtype = np.dtype(getattr(np, dtype))

    # Open the output file
    with h5py.File(output_file, 'w') as f:
        # Loop over the directories in the input directory
        f.create_group
        progress = tqdm(subs)
        for subject in progress:

            f.attrs['volume_size'] = volume_size
            f.attrs['nb_coeffs'] = nb_coeffs
            f.attrs['bundles'] = bundles

            sub_id = split(subject)[-1]
            progress.set_description(sub_id)
            # Get the path to the FODF volume
            fodf_file = join(
                subject, f'{sub_id}__fodf.nii.gz')
            # Get the path to the WM mask
            mask_wm_file = join(
                subject, f'{sub_id}__mask_wm.nii.gz')

            # Load the FODF volume
            fodf_img = nib.load(fodf_file)

            # Resampling volume
            resampled_img = resample_volume(fodf_img, ref_img=None,
                                            volume_shape=[volume_size],
                                            iso_min=False,
                                            voxel_res=None,
                                            interp='lin',
                                            enforce_dimensions=False)

            # Load the WM mask
            mask_wm_img = nib.load(mask_wm_file)

            # Resampling WM mask
            resampled_mask_wm = resample_volume(mask_wm_img, ref_img=None,
                                                volume_shape=[volume_size],
                                                iso_min=False,
                                                voxel_res=None,
                                                interp='nn',
                                                enforce_dimensions=False)

            fodf_data = resampled_img.get_fdata().transpose((3, 0, 1, 2))
            mask_wm_data = resampled_mask_wm.get_fdata()[None, ...]

            affine = resampled_img.affine
            assert np.allclose(affine, resampled_mask_wm.affine, atol=1e-5), (
                sub_id, np.abs(affine - resampled_mask_wm.affine))

            # Create a group for the subject
            group = f.create_group(sub_id)

            # Create a dataset for the FODF volume
            group.create_dataset('fodf', data=fodf_data[:nb_coeffs, ...],
                                 compression="gzip", dtype=out_dtype)

            # Create a dataset for the FODF volume
            group.create_dataset('mask_wm', data=mask_wm_data,
                                 compression="gzip", dtype=out_dtype)

            # Create a dataset for the affine transformation matrix
            group.create_dataset('affine', data=affine)

            # Add bundles to dataset
            bundle_group = group.create_group('bundles')

            for b in bundles:
                # name = sub_id + '__' + b
                label_file = join(subject, 'labels', b + '.nii.gz')
                try:
                    b_label_img = nib.load(label_file)
                except FileNotFoundError:
                    print('{} does not have a {}'.format(sub_id, b))
                    continue
                reshaped_b_label = apply_transform(np.eye(4), mask_wm_img,
                                                   b_label_img,
                                                   interp='nearest',
                                                   keep_dtype=True)
                # Resampling WM mask
                resampled_b_label = resample_volume(reshaped_b_label,
                                                    ref_img=None,
                                                    volume_shape=[volume_size],
                                                    iso_min=False,
                                                    voxel_res=None,
                                                    interp='nn',
                                                    enforce_dimensions=False)

                # Group for the bundle
                bundle = bundle_group.create_group(b)

                bundle.create_dataset(
                    'labels', data=resampled_b_label.get_fdata(),
                    compression="gzip", dtype=out_dtype)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subs', nargs='+', type=str,
                        help='Input directory containing FODF volumes')
    parser.add_argument('output_file', help='Output HDF5 file')
    parser.add_argument('--volume_size', default=128, type=int,
                        help='Volume size to resample to.')
    parser.add_argument('--sh_order', type=int, default=6,
                        choices=[2, 4, 6, 8],
                        help='SH order to use.')
    parser.add_argument('--dtype', choices=['float16', 'float32'],
                        default='float32', type=str,
                        help='Cast data to this type before storing.')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite output file if it exists.')
    parser.add_argument('--bundles', type=str, nargs='+',
                        default=TRACTSEG_BUNDLES)
    args = parser.parse_args()

    # Check if the input directory exists
    for s in args.subs:
        if not exists(s):
            raise ValueError('Input directory does not exist')

    # Check if the output file exists
    if exists(args.output_file) and not args.force:
        raise ValueError('Output file already exists')

    n_coefs = int(
            (args.sh_order + 2) * (args.sh_order + 1) // 2)

    # Create the dataset
    create_dataset(args.subs, args.output_file, args.volume_size,
                   n_coefs, args.dtype, args.bundles)


if __name__ == '__main__':
    main()
