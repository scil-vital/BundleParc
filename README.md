# LabelSeg
Code and pretrained model for LabelSeg: automatic tract labeling without tractography

## Installation

Only Python3.10 is currently supported. It is recommended to install the software in a [virtualenv](https://virtualenv.pypa.io/en/latest/).

To install, in the cloned project's folder:

```bash
pip install -e .
```

Docker containers are coming soon-ish.

## Running

To run LabelSeg, use

```labelseg_predict```

The software will output 71 files, each corresponding to a bundle's label map. The bundle definitions follow TractSeg's, minus the whole CC.

## Troubleshooting

Ran into a problem during installation or prediction ? Have a question ? Please open an issue !
