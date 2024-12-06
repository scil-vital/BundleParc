#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" Create a dataset for the LabelSeg model. The dataset includes, for each
subject, the FODF volume, the WM mask, the affine transformation matrix, and
the tracts.
"""

import argparse

import h5py
import nibabel as nib
import numpy as np

from os.path import exists, join, split

from tqdm import tqdm

from scilpy.image.volume_operations import resample_volume, apply_transform


def create_dataset(
    subs, output_file, volume_size, nb_coeffs, nb_points, dtype, bundles
):
    """
    """

    out_dtype = np.dtype(getattr(np, dtype))

    # Open the output file
    with h5py.File(output_file, 'w') as f:
        # Loop over the directories in the input directory
        progress = tqdm(subs)
        for subject in progress:

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
                name = sub_id + '__' + b
                mask_file = join(subject, 'bundles', name + '.nii.gz')
                end_file = join(subject, 'bundles', name + '_endpoints.nii.gz')
                try:
                    b_mask_img = nib.load(mask_file)
                    b_end_img = nib.load(end_file)
                except FileNotFoundError:
                    print('{} does not have a {}'.format(sub_id, b))
                    continue
                reshaped_b_mask = apply_transform(np.eye(4), mask_wm_img, b_mask_img,
                                                  interp='nearest',
                                                  keep_dtype=True)
                reshaped_b_end = apply_transform(np.eye(4), mask_wm_img, b_end_img,
                                                 interp='nearest',
                                                 keep_dtype=True)

                # Resampling WM mask
                resampled_b_mask = resample_volume(reshaped_b_mask,
                                                   ref_img=None,
                                                   volume_shape=[volume_size],
                                                   iso_min=False,
                                                   voxel_res=None,
                                                   interp='nn',
                                                   enforce_dimensions=False)
                resampled_b_end = resample_volume(reshaped_b_end, ref_img=None,
                                                  volume_shape=[volume_size],
                                                  iso_min=False,
                                                  voxel_res=None,
                                                  interp='nn',
                                                  enforce_dimensions=False)

                # Group for the bundle
                bundle = bundle_group.create_group(name)

                bundle.create_dataset(
                    'mask', data=resampled_b_mask.get_fdata(),
                    compression="gzip", dtype=out_dtype)

                bundle.create_dataset(
                    'endpoints', data=resampled_b_end.get_fdata(),
                    compression="gzip", dtype=out_dtype)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subs', nargs='+', type=str,
                        help='Input directory containing FODF volumes')
    parser.add_argument('output_file', help='Output HDF5 file')
    parser.add_argument('--volume_size', default=96, type=int,
                        help='Volume size to resample.')
    parser.add_argument('--nb_coeffs', choices=[28, 45],
                        default=28, type=int,
                        help='Nb. of SH coeffs to use.')
    parser.add_argument('--nb_points', type=int, default=64,
                        help='Nb. of points to resample the streamlines to.')
    parser.add_argument('--dtype', choices=['float16', 'float32'],
                        default='float32', type=str,
                        help='Cast data to this type before storing.')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite output file if it exists.')
    parser.add_argument('--bundles', type=str, nargs='+',
                        default=['AF_L', 'AF_R', 'CC_Fr_1', 'CC_Fr_2', 'CC_Oc',
                                 'CC_Pa', 'CC_Pr_Po', 'CG_L', 'CG_R', 'FAT_L',
                                 'FAT_R', 'FPT_L', 'FPT_R', 'FX_L', 'FX_R',
                                 'IFOF_L', 'IFOF_R', 'ILF_L', 'ILF_R', 'MCP',
                                 'MdLF_L', 'MdLF_R', 'OR_ML_L', 'OR_ML_R',
                                 'POPT_L', 'POPT_R', 'PYT_L', 'PYT_R', 'SLF_L',
                                 'SLF_R', 'UF_L', 'UF_R'])
    args = parser.parse_args()

    # Check if the input directory exists
    for s in args.subs:
        if not exists(s):
            raise ValueError('Input directory does not exist')

    # Check if the output file exists
    if exists(args.output_file) and not args.force:
        raise ValueError('Output file already exists')

    # Create the dataset
    create_dataset(args.subs, args.output_file, args.volume_size,
                   args.nb_coeffs, args.nb_points, args.dtype, args.bundles)


if __name__ == '__main__':
    main()
