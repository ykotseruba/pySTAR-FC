# pySTAR-FC
pySTAR-FC is an application for predicting human fixation locations on arbitrary static images.

This is a Python re-implementation of [STAR-FC](https://github.com/TsotsosLab/STAR-FC) published in CVPR'18 paper ["Active Fixation Control to Predict Saccade Sequences"](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wloka_Active_Fixation_Control_CVPR_2018_paper.pdf). Note that the Python version uses a faster OpenGL-based foveation therefore it produces the results that are *different from the ones in the paper*.

![pySTAR-FC in action](examples/Yarbus.gif)

## Getting Started

### Installation

This setup was tested with NVIDIA Titan X on Ubuntu 22.04 with Python 3.9.

#### Docker

Install docker following the instructions [here](https://docs.docker.com/engine/install/ubuntu/).

<!-- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) following the instructions in the official repository. There are also good resources elsewhere that describe Docker installation in more detail, for example [this one for Ubuntu 16.04](https://chunml.github.io/ChunML.github.io/project/Installing-NVIDIA-Docker-On-Ubuntu-16.04/). -->

Add your name to the docker group so you can run docker commands without sudo:
```
usermod -aG docker <yourLoginUsername>
```

After Docker is installed all you need to do is to build a container using the scripts in the ```docker``` folder (may take 30+ mins):
```
cd pySTAR-FC
sh docker/build_docker.sh
```

NOTE: To use OpenGL, you will need to install the same GPU driver as in your system inside the container. The ```docker/build_docker.sh``` script should handle this automatically. But if it fails, do the following:

1. Find your GPU driver version (run either `nvidia-smi` or ```DRIVER_VERSION=$(glxinfo | grep "OpenGL version string" | rev | cut -d" " -f1 | rev) ```). 

2. Get the link for the ```.run``` file from [NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=en-us) and download the driver.

3. Place the `.run` file into the ```docker``` directory and rename the file to ```NVIDIA-DRIVER.run```.

4. Run `docker/build_docker.sh` again.


### Virtual environment

Apt-get dependencies:

```
sudo apt-get install -y \
    libglib2.0-dev libsm6 python3-pyqt5 \
    libglfw3-dev mesa-utils kmod
```

Python dependencies:

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r docker/requirements.txt

```

Depending on your system, you might need to install CUDA (v11.7.8) (https://developer.nvidia.com/cuda-toolkit-archive) and CuDNN (v8) (https://developer.nvidia.com/rdp/cudnn-archive).

<!-- pip3 install pycuda==2017.1.1 -->


<!-- Install [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit-archive), [TensorFlow](https://www.tensorflow.org/install/), [CuDNN 6.0](https://developer.nvidia.com/rdp/cudnn-archive) for CUDA 8.0 ([installation instructions](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)). -->


<!-- install pyCUDA library:

If you are getting 'pycuda._driver.Error: cuInit failed: unknown error' when running the code, try rebooting the machine
 -->

### Additional dependencies

pySTAR-FC relies on several saliency models that also need to be installed.

Tensorflow versions of DeepGazeII and ICF are no longer available from [https://deepgaze.bethgelab.org/] and the webpage is not accessible anymore. 

1. Download the files from mirror at https://drive.google.com/file/d/1e2ktks8XOsjGWotFpE4GtXJNIjZxCyVd/view?usp=drive_link 

2. Place checkpoint files (```ckpt.data```, ```ckpt.index``` and ```ckpt.meta```) into ```pySTAR_FC/contrib/DeepGazeII``` and ```pySTAR_FC/contrib/ICF``` folders respectively. 

3. Copy ```centerbias.npy``` file into both `ICF` and `DeepGazeII` folders.

<!-- Download SALICONtf from [https://github.com/ykotseruba/SALICONtf] and place the files it in ```pySTAR-FC/contrib/SALICONtf```. Download pre-trained SALICONtf weights:
```
cd contrib/SALICONtf/models
sh download_pretrained_weights.sh
sh download_vgg_weights.sh
``` -->

## Running STAR-FC

Below are instructions on how to run a demo of STAR-FC on a single image (```images/Yarbus_scaled.jpg```).

If using Docker:
```
sh docker_scripts/run.sh 
```
Then inside Docker:
```
python3 src/STAR_FC.py -v -c config_files/test.ini
```

Without Docker simply run:
```
python3 src/STAR_FC.py -v -c config_files/test.ini
```

There are only two command line options available:
* -v for visualization (optional)
* -c for config file in .ini format

All internal parameters of the algorihtm are set via configuration file (for available options and purpose of each parameter see example config file `config_files/template_config.ini`).

Should you have any questions, feel free to raise an issue or email yulia84@yorku.ca.


### Authors

* **Iuliia Kotseruba** - *Python version of the code*
* **Calden Wloka** - *theory, original C++ implementation for TarzaNN*

### Citing us

If you find our work useful in your research, please consider citing:

```latex
@InProceedings{Wloka_CVPR18,
  author = {Wloka, Calden and Kotseruba, Iuliia and Tsotsos, J. K.},
  title = {Saccade Sequence Prediction: Beyond Static Saliency Maps},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```
