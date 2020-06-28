#!/bin/bash

#Check driver version and update the DRIVER LINK
DRIVER_VERSION=$(glxinfo | grep "OpenGL version string" | rev | cut -d" " -f1 | rev) 

DRIVER_LINK="http://us.download.nvidia.com/XFree86/Linux-x86_64/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run"
if [ ! -f NVIDIA-DRIVER.run ]; then
    wget $DRIVER_LINK 
    mv `basename $DRIVER_LINK` NVIDIA-DRIVER.run
fi

PROJECT_ROOT="$(cd "$(dirname "$0")"; cd ..; pwd)"
echo "Building starfcpy Docker image..."
docker build --no-cache -t starfcpy .

