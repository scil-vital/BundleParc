import argparse

import h5py
import nibabel as nib
import numpy as np

from os.path import exists, join, split
from tqdm import tqdm

from scilpy.image.volume_operations import resample_volume

""" Create a dataset of FODF volume in the form of a HDF5 file.
Each subject is represented by a 4D array of FODF volumes with their
corresponding affine transformation matrix.
"""


def create_dataset(subs, output_file, volume_size, nb_coeffs, dtype):
    """ Each subject is contained in a separate directory. Each directory
    contains a FODF volume in NIfTI format. The FODF volume is a 4D array
    with the last dimension representing the number of spherical harmonics
    coefficients. The affine transformation matrix is stored in the header
    of the NIfTI file.
    """

    out_dtype = np.dtype(getattr(np, dtype))

    # Open the output file
    with h5py.File(output_file, 'w') as f:
        # Loop over the directories in the input directory
        progress = tqdm(subs)
        for subject in progress:

            sub_id = split(subject)[-1]
            progress.set_description(sub_id)

            fodf_file = join(
                subject, 'FODF', f'{sub_id}__fodf.nii.gz')

            # Load the FODF volume
            fodf_img = nib.load(fodf_file)

            # Resampling volume
            resampled_img = resample_volume(fodf_img, ref_img=None,
                                            volume_shape=[volume_size],
                                            iso_min=False,
                                            voxel_res=None,
                                            interp='lin',
                                            enforce_dimensions=False)

            fodf_data = resampled_img.get_fdata().transpose((3, 0, 1, 2))

            affine = resampled_img.affine

            # Create a group for the subject
            group = f.create_group(sub_id)

            # Create a dataset for the FODF volume
            group.create_dataset('fodf', data=fodf_data[:nb_coeffs, ...],
                                 compression="gzip", dtype=out_dtype)
            # Create a dataset for the affine transformation matrix
            group.create_dataset('affine', data=affine)


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
                   args.nb_coeffs, args.dtype)


if __name__ == '__main__':
    main()
