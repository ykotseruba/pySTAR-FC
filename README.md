# pySTAR-FC
pySTAR-FC is an application for predicting human fixation locations on arbitrary static images.
More details on the algorithm and results can be found in the paper ["Saccade Sequence Prediction: Beyond Static Saliency Maps"](https://arxiv.org/pdf/1711.10959.pdf).

## Getting Started

## Docker installation (recommended)

We recommend using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run everything inside a container.

Add your name to the docker group so you can run docker commands without sudo
```
usermod -aG docker <yourLoginUsername>
```

To build the container:

```
cd STAR-FCpy/
sudo script/build
```

And to run the container:

```
sudo script/run -v -c <path_to_config_file>
```
There are only two command line options:
* -v for visualization  (optional)
* -c for config file in .ini format

The code and files are mounted in `/opt/STAR-FC`, which you can edit from your host machine using your usual editor or from within the container. Remember that any files created within the container will belong to root, but there is no harm in `chown`ing them back to your host user.

If the `nvidia-docker` install fails due to docker version mismatch, install the latest stable `docker-ce` from [the official repos](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and try again.


### Manual installation
* python3
* CUDA
* pyCUDA
* TensorFlow
* numpy
* scipy
* matplotlib
* cv2
* DeepGazeII
* ICF
* SALICONtf

```
pip3 install numpy scipy matplotlib opencv-python pycuda tensorflow-gpu==1.4.0
```

Install [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit-archive), [TensorFlow](https://www.tensorflow.org/install/), [CuDNN 6.0](https://developer.nvidia.com/rdp/cudnn-archive) for CUDA 8.0 ([installation instructions](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)).

Download DeepGazeII and ICF models from [https://deepgaze.bethgelab.org/] and place the files into pySTAR_FC/contrib/DeepGazeII and pySTAR_FC/contrib/ICF folders respectively (only the checkpoint files and centerbias for each model).

install pyCUDA library:

If you are getting 'pycuda._driver.Error: cuInit failed: unknown error' when running the code, try rebooting the machine

### Running STAR-FC

The following command will run a demo of STAR-FC on a single image (Yarbus_scaled.jpg in images/ folder):
```
python3 src/STAR_FC.py -v -c config_files/test.ini
```

There are only two command line options:
* -v for visualization (optional)
* -c for config file in .ini format

All internal parameters of the algorihtm are set via configuration file (for available options and purpose of each parameter see example config file `config_files/template_config.ini`).


### Authors

* **Iuliia Kotseruba** - *Python version of the code*
* **Calden Wloka** - *theory, original C++ implementation for TarzaNN*

### Citing us

If you find our work useful in your research, please consider citing:

```latex
@article{wloka2017saccade,
  title={Saccade Sequence Prediction: Beyond Static Saliency Maps},
  author={Wloka, Calden and Kotseruba, Iuliia and Tsotsos, John K},
  journal={arXiv preprint arXiv:1711.10959},
  year={2017}
}
```