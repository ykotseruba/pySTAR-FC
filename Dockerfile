FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
#FROM tensorflow/tensorflow:1.13.0rc1-gpu-py3

ENV STARFCPY_ROOT="/opt/STAR-FC"

################################################################################
# Prerequisites
################################################################################

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3-pip python3-pillow python3-tk \
    libglib2.0-dev libsm6 \
    git vim silversearcher-ag tmux

COPY requirements.txt STARFCPY_ROOT/
WORKDIR STARFCPY_ROOT

# HACK: should install from requirements.txt file instead.
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install pycuda==2017.1.1

ENV PYTHONPATH=$STARFCPY_ROOT/contrib/SALICONtf/src/:$PYTHONPATH
WORKDIR $STARFCPY_ROOT

