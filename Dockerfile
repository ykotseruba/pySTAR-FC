FROM nvidia/cuda:8.0-cudnn6-devel
#FROM tensorflow/tensorflow:1.13.0rc1-gpu-py3

################################################################################
# Prerequisites
################################################################################

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3-pip python3-pillow python3-tk \
    libglib2.0-dev libsm6 \
    git vim silversearcher-ag tmux

# HACK: should install from requirements.txt file instead.
RUN pip3 install --upgrade pip
RUN pip3 install numpy scipy matplotlib opencv-python 
RUn pip3 install pycuda tensorflow-gpu==1.13.0rc1

ENV STARFCPY_ROOT="/opt/STAR-FC"
ENV PYTHONPATH=$STARFCPY_ROOT/contrib/SALICONtf/src/:$PYTHONPATH
WORKDIR $STARFCPY_ROOT

