FROM nvidia/cuda:8.0-cudnn6-devel

################################################################################
# Prerequisites
################################################################################

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3-pip python3-pillow python3-tk \
    libglib2.0-dev libsm6 \
    git vim silversearcher-ag tmux

# HACK: should install from requirements.txt file instead.
RUN pip3 install --upgrade pip
RUN pip3 install numpy scipy matplotlib opencv-python && \
    pip3 install pycuda tensorflow-gpu==1.4.0

ENV STARFCPY_ROOT="/opt/STAR-FC"
WORKDIR $STARFCPY_ROOT
#RUN git clone https://github.com/NVIDIA/nccl.git && cd nccl && make && make install && cd .. && rm -rf nccl

WORKDIR /workspace
