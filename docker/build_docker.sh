#!/bin/bash
readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}")")
IMAGE_NAME=base_images/cuda
DOCKER_FILENAME=Dockerfile
TAG=pystarfc


while [[ $# -gt 0 ]]
do key="$1"

case $key in
	-im|--image_name)
	IMAGE_NAME="$2"
	shift # past argument
	shift # past value
	;;
	-t|--tag)
	TAG="$2"
	shift # past argument
	shift # past value
	;;
	# -f|--file)
	# DOCKER_FILENAME="$2"
	# shift # past argument
	# shift # past value
	# ;;	
	-h|--help)
	shift # past argument
	echo "Options:"
	echo "	-im, --image_name	name of the docker image (default \"base_images/tensorflow\")"
	echo "	-t, --tag		image tag name (default \"tf2.2-gpu\")"
	# echo "	-f, --file		docker file name (default \"Dockerfile_tf2\")"
	exit
	;;
	*)
	echo " Wrong option(s) is selected. Use -h, --help for more information "
	exit
	;;
esac
done

#Check driver version and update the DRIVER LINK
DRIVER_VERSION=$(glxinfo | grep "OpenGL version string" | rev | cut -d" " -f1 | rev) 

DRIVER_LINK="http://us.download.nvidia.com/XFree86/Linux-x86_64/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run"
if [ ! -f NVIDIA-DRIVER.run ]; then
    wget $DRIVER_LINK 
    mv `basename $DRIVER_LINK` NVIDIA-DRIVER.run
fi


docker buildx build -t ${IMAGE_NAME}:${TAG} \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
	-f ${SCRIPT_DIR}/${DOCKER_FILENAME} \
	${SCRIPT_DIR}
