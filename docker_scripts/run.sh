#!/bin/bash

# setup x auth environment for visual support
XAUTH=$(mktemp /tmp/.docker.xauth.XXXXXXXXX)
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

PROJECT_ROOT="$(cd "$(dirname "$0")"; cd ..; pwd)"
STARFCPY_ROOT="/opt/STAR-FC"


usage() {
    echo "Usage: $0 [-v] [-g 0] -c <config_file_path>"
    echo "Options:"
    echo "-v \t visualization on"
    echo "-c \t path to config file with extension .ini (see config_files for examples)"
    echo "-g \t which GPU to run on (default 0)"
}

vis_flag=''
config_file_path=''

GPU_DEVICE=0

while getopts "h?vc:g:" opt; do
    case "$opt" in
        h|\?)
            usage
            exit 0
            ;;
        v)  vis_flag='-v'
            ;;
        c)  config_file_path=$OPTARG
            ;;
        g)  GPU_DEVICE=$OPTARG
        esac
done
shift "$((OPTIND-1))"

if [ -z "$config_file_path" ]; then
    echo "ERROR: config file not provided!"
    usage
    exit 1
fi

# -v /tmp/.X11-unix:/tmp/.X11-unix \
xhost +local:starfcpy

nvidia-docker run -it \
  --gpus "device=${GPU_DEVICE}" \
  --name starfcpy \
  -h starfcpy \
  -v ${PROJECT_ROOT}:${STARFCPY_ROOT} \
  -v /dev/input \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$XAUTH \
  -v $XAUTH:$XAUTH \
  -env="DISPLAY" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --net=host \
  -w ${STARFCPY_ROOT} \
  --rm \
  starfcpy python3 src/STAR_FC.py $vis_flag -c $config_file_path
xhost -local:starfcpy
