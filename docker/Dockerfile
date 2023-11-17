#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
#FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04


## Setup user id inside docker

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID docker_user

## Install dependencies

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3-pip \
    libglib2.0-dev libsm6 \
    git wget \
    libglfw3-dev mesa-utils kmod
    
RUN apt-get install -y binutils

#version="$(glxinfo | grep "OpenGL version string" | rev | cut -d" " -f1 | rev)"

COPY NVIDIA-DRIVER.run /tmp/NVIDIA-DRIVER.run
RUN sh /tmp/NVIDIA-DRIVER.run -a -N --ui=none --no-kernel-module
RUN rm /tmp/NVIDIA-DRIVER.run

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt


USER docker_user
