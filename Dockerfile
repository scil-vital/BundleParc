FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git build-essential gcc g++ \
      libblas-dev liblapack-dev libgl1 libxrender1 \
      libfreetype6-dev pkg-config software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get -y install \
  python3.10-dev python3-pip python3.10-venv

RUN git clone https://github.com/scil-vital/BundleParc.git

WORKDIR /BundleParc

ENV VIRTUAL_ENV=/opt/venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN mkdir checkpoints
ADD https://zenodo.org/records/15579498/files/123_4_5_bundleparc.ckpt checkpoints/123_4_5_bundleparc.ckpt

RUN pip3 install --upgrade pip
RUN pip install Cython numpy packaging
RUN SETUPTOOLS_USE_DISTUTILS=stdlib pip install -e .
