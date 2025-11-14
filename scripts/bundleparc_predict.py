#!/usr/bin/env python

"""
BundleParc: automatic tract labelling without tractography.

This method takes as input fODF maps and outputs 71 bundle label maps. These maps can then be used to perform tractometry/tract profiling/radiomics. The bundle definitions follow TractSeg's minus the whole CC.

**IMPORTANT**: fODF inputs are presumed to come from Tractoflow, must have stride of -1,2,3,4 and must be BET and cropped. fODFs must be in SH format, basis descoteaux07_legacy and can be of order < 8 but accuracy may be reduced.

Model weights will be downloaded the first time the script is run, which will require an internet connection at runtime. Otherwise they can be manually downloaded from zenodo [1] and by specifying --checkpoint.

Example usage:
    $ scil_fodf_bundleparc fodf.nii.gz --out_prefix sub-001__

Example output:
    sub-001__AF_left.nii.gz, sub-001__AF_right.nii.gz, ..., sub-001__UF_right.nii.gz

The output can be further processed with scil_bundle_mean_std to compute statistics for each bundle.

The default value of 50 for --min_blob_size was found empirically on adult brains at a resolution of 1mm^3. The best value for your dataset may differ.

This script requires a GPU with ~6GB of available memory. If you use half-precision (float16) inference, you may be able to run it with ~3GB of GPU memory available. Otherwise, install the CPU version of PyTorch. Execution on MacOS is not supported for now.

Parts of the implementation are based on or lifted from:
    SAM-Med3D: https://github.com/uni-medical/SAM-Med3D
    Multidimensional Positional Encoding: https://github.com/tatp22/multidim-positional-encoding

To cite: Antoine Théberge, Zineb El Yamani, François Rheault, Maxime Descoteaux, Pierre-Marc Jodoin (2025). LabelSeg. ISMRM Workshop on 40 Years of Diffusion: Past, Present & Future Perspectives, Kyoto, Japan.

[1]: Descoteaux, M., Deriche, R., Knösche, T. R., & Anwander, A. (2007). Deterministic and probabilistic tractography based on complex fibre orientation distributions. IEEE Transactions on Medical Imaging, 26(11), 1464-1477.
[2]: https://zenodo.org/records/15579498
"""  # noqa

import argparse
import logging
import nibabel as nib
import numpy as np
import os

from argparse import RawTextHelpFormatter

from scilpy.io.utils import (
    assert_inputs_exist, assert_output_dirs_exist_and_empty,
    add_overwrite_arg, add_verbose_arg)
from scilpy.image.volume_operations import resample_volume

from scilpy.ml.utils import get_device, IMPORT_ERROR_MSG
from scilpy import SCILPY_HOME

from BundleParc.predict import predict
from BundleParc.models.utils import get_model

from dipy.utils.optpkg import optional_package
torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)

DEFAULT_CKPT = os.path.join(SCILPY_HOME, 'checkpoints', 'bundleparc.ckpt')
DEFAULT_BUNDLES = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right', 'MCP', 'MLF_left', 'MLF_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right', 'SLF_III_left', 'SLF_III_right', 'SLF_II_left', 'SLF_II_right', 'SLF_I_left', 'SLF_I_right', 'STR_left', 'STR_right', 'ST_FO_left', 'ST_FO_right', 'ST_OCC_left', 'ST_OCC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'T_OCC_left', 'T_OCC_right', 'T_PAR_left', 'T_PAR_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PREC_left', 'T_PREC_right', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left', 'T_PREM_right', 'UF_left', 'UF_right']  # noqa E501


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__ + '\n' + IMPORT_ERROR_MSG,
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('in_fodf',
                        help='Input fODF volume in nifti format. ')
    parser.add_argument('--out_prefix', default='',
                        help='Output file prefix. Default is nothing. ')
    parser.add_argument('--out_folder', default='bundleparc',
                        help='Output destination. Default is [%(default)s].')
    parser.add_argument('--volume_size', type=int, default=128,
                        help='Resolution to resample to for inference. '
                             'Default is [%(default)s]^3.')
    parser.add_argument('--half_precision', action='store_true',
                        help='Use half precision (float16) for inference. '
                             'This reduces memory usage but may lead to '
                             'reduced accuracy.')
    parser.add_argument('--checkpoint', default=DEFAULT_CKPT, required=True,
                        help='Checkpoint (.ckpt) containing hyperparameters '
                             'and weights of model. Default is '
                             '[%(default)s]. If the file does not exist, it '
                             'will be downloaded.')
    blob_group = parser.add_mutually_exclusive_group()
    blob_group.add_argument('--min_blob_size', type=int, default=50,
                            help='Minimum blob size (in voxels) to keep. '
                                 'Smaller blobs will be removed. Default is '
                                 '[%(default)s].')
    blob_group.add_argument('--keep_biggest_blob', action='store_true',
                            help='Only keep the biggest blob predicted.')

    add_overwrite_arg(parser)
    add_verbose_arg(parser)

    return parser


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_fodf], [])
    assert_output_dirs_exist_and_empty(parser, args, args.out_folder,
                                       create_dir=True)

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    device = get_device()
    # Load the model
    model, bundles = get_model(args.checkpoint, {'pretrained': True})
    model.to(device)

    fodf_in = nib.load(args.in_fodf)
    X, Y, Z, C = fodf_in.get_fdata(dtype=np.float32).shape

    # TODO in future release: infer these from model
    n_coefs = 45
    img_size = args.volume_size

    # Check the number of coefficients in the input fODF
    if C < n_coefs:
        logging.warning(f'Input fODFs have fewer than {n_coefs} coefficients. '
                        'Accuracy may be reduced.')
    if C > n_coefs:
        logging.warning(f'Input fODFs have more than {n_coefs} coefficients. '
                        f'Only the first {n_coefs} will be used.')

    # Resampling volume to fit the model's input at training time
    resampled_img = resample_volume(fodf_in, ref_img=None,
                                    volume_shape=[img_size],
                                    iso_min=False,
                                    voxel_res=None,
                                    interp='lin',
                                    enforce_dimensions=False)

    # # Get the voxel size of the input fODF after resampling
    # # Presuming isotropic resampling
    # voxel_size = np.mean(resampled_img.header.get_zooms()[:3])

    # Predict label maps. `predict` is a generator
    # yielding one label map per bundle and its name.
    for y_hat_label, b_name in predict(
        model,
        resampled_img.get_fdata(dtype=np.float32),
        n_coefs,
        bundles,
        args.min_blob_size,
        args.keep_biggest_blob,
        half_precision=args.half_precision,
        verbose=logging.getLogger().getEffectiveLevel() < logging.WARNING
    ):
        # Format the output as a nifti image
        label_img = nib.Nifti1Image(y_hat_label,
                                    resampled_img.affine,
                                    resampled_img.header,
                                    dtype=y_hat_label.dtype)

        # Resampling volume to fit the original image size
        resampled_label = resample_volume(label_img, ref_img=None,
                                          volume_shape=[X, Y, Z],
                                          iso_min=False,
                                          voxel_res=None,
                                          interp='nn',
                                          enforce_dimensions=False)
        # Save it
        nib.save(resampled_label, os.path.join(
            args.out_folder, f'{args.out_prefix}{b_name}.nii.gz'))


if __name__ == "__main__":
    main()
