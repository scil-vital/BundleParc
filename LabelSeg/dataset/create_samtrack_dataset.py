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

from os import listdir
from os.path import exists, isfile, join, split

from tqdm import tqdm

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.tracking.streamline import set_number_of_points
from scilpy.image.volume_operations import resample_volume
from scilpy.io.streamlines import load_tractogram


def create_dataset(
    subs, output_file, volume_size, nb_coeffs, nb_points, dtype
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
                subject, 'fodf', f'{sub_id}__fodf.nii.gz')
            # Get the path to the WM mask
            mask_wm_file = join(
                subject, 'mask', f'{sub_id}__mask_wm.nii.gz')

            # Get the path to the tracts
            tracts_folder = join(
                subject, 'tractography')
            tract_files = [join(tracts_folder, f) for f in listdir(
                tracts_folder) if isfile(join(tracts_folder, f))]
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

            # Load the tracts as a stateful tractogram
            all_streamlines = []
            for tract_file in tract_files:
                tractogram = load_tractogram(
                    tract_file, fodf_img, to_space=Space.RASMM)
                streamlines = set_number_of_points(
                    tractogram.streamlines, nb_points)
                all_streamlines.extend(streamlines)
            all_streamlines_array = np.asarray(all_streamlines)

            new_sft = StatefulTractogram(
                all_streamlines_array, resampled_img, space=Space.RASMM)

            before_len = len(new_sft.streamlines)

            new_sft.remove_invalid_streamlines()

            after_len = len(new_sft.streamlines)

            new_sft.to_vox()

            # Create a dataset for the tracts
            group.create_dataset(
                'tracts', data=np.asarray(new_sft.streamlines),
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
                   args.nb_coeffs, args.nb_points, args.dtype)


if __name__ == '__main__':
    main()
