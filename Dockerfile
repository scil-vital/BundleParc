FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git build-essential \
      liblapack-dev libopenblas-dev libgl1 libxrender1

RUN git git@github.com:scil-vital/BundleParc.git

WORKDIR /BundleParc

RUN mkdir checkpoints
ADD https://zenodo.org/records/15579498/files/123_4_5_bundleparc.ckpt checkpoints/123_4_5_bundleparc.ckpt

RUN pip install Cython numpy packaging
RUN pip install -e .
