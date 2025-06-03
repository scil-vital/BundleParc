
import argparse
import os
import torch

from argparse import RawTextHelpFormatter
from pathlib import Path

from scilpy.io.utils import (
    add_overwrite_arg)

from BundleParc.models.bundleparc import BundleParc


def _build_arg_parser(parser):
    parser.add_argument('checkpoint', type=str,
                        help='Checkpoint')
    parser.add_argument('destination', type=str,
                        help='Checkpoint')
    add_overwrite_arg(parser)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter)

    _build_arg_parser(parser)
    args = parser.parse_args()

    return parser, args


def main():

    parser, args = parse_args()

    # Load the model's hyper and actual params from a saved checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
    except RuntimeError:
        # If the model was saved on a GPU and is being loaded on a CPU
        # we need to specify map_location=torch.device('cpu')
        checkpoint = torch.load(
            args.checkpoint, map_location=torch.device('cpu'),
            weights_only=False)

    # The model's class is saved in hparams
    models = {
        # Add other architectures here
        'BundleParc': BundleParc,
        'LabelSeg': BundleParc,
    }
    # TODO: investigate why hparams are not in checkpoint
    hyper_parameters = checkpoint

    print(hyper_parameters.keys())
    print(hyper_parameters['hyper_parameters'].keys())

    # Rename the keys in the state_dict, by replacing "labelseg" with "bundleparc"
    new_state_dict = {}
    for key, value in hyper_parameters['state_dict'].items():
        new_key = key.replace('labelseg', 'bundleparc')
        new_state_dict[new_key] = value

    hyper_parameters['state_dict'] = new_state_dict

    # Rename the hyper_parameters key name
    hyper_parameters['hyper_parameters']['name'] = 'BundleParc'

    # Save the modified checkpoint
    torch.save(
        hyper_parameters, args.destination)


if __name__ == "__main__":
    main()
