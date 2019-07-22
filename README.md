# pySTAR-FC
pySTAR-FC is an application for predicting human fixation locations on arbitrary static images.
More details on the algorithm and results can be found in our CVPR'18 paper ["Active Fixation Control to Predict Saccade Sequences"](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wloka_Active_Fixation_Control_CVPR_2018_paper.pdf).

![pySTAR-FC in action](examples/Yarbus.gif)

## Getting Started

### Installation

We tested this setup with NVIDIA Titan X on Ubuntu 16.04 with Python 3.5.

#### Docker (strongly recommended)

Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) following the instructions in the official repository. There are also good resources elsewhere that describe Docker installation in more detail, for example [this one for Ubuntu 16.04](https://chunml.github.io/ChunML.github.io/project/Installing-NVIDIA-Docker-On-Ubuntu-16.04/).

Add your name to the docker group so you can run docker commands without sudo:
```
usermod -aG docker <yourLoginUsername>
```

After Docker is installed all you need to do is to build a container using the scripts in the ```docker_scripts``` folder:
```
cd pySTAR-FC
sh docker_scripts/build.sh
```

NOTE: You will need to install the same GPU driver as in your system inside the container. ```docker_scripts/build.sh``` outputs the version of the driver on your system. Obtain the link for the ```.run``` file from [NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=en-us) and replace the ```DRIVER_LINK``` in ```build.sh```. Or download the driver manually, place it into ```pySTAR-FC``` directory and rename the file to ```NVIDIA-DRIVER.run```.

To run the container:

```
sudo script/run -v -c <path_to_config_file>
```
There are only two command line options:
* -v for visualization  (optional)
* -c for config file in .ini format

The code and files are mounted in `/opt/STAR-FC`, which you can edit from your host machine using your usual editor or from within the container. Remember that any files created within the container will belong to root, but there is no harm in `chown`ing them back to your host user.

### Manual installation

```
pip3 install -r requirements.txt
```
<!-- pip3 install pycuda==2017.1.1 -->


<!-- Install [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit-archive), [TensorFlow](https://www.tensorflow.org/install/), [CuDNN 6.0](https://developer.nvidia.com/rdp/cudnn-archive) for CUDA 8.0 ([installation instructions](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)). -->


<!-- install pyCUDA library:

If you are getting 'pycuda._driver.Error: cuInit failed: unknown error' when running the code, try rebooting the machine
 -->

### Additional dependencies

pySTAR-FC relies on several saliency models that also need to be installed.

Download DeepGazeII and ICF models from [https://deepgaze.bethgelab.org/] and place the files into ```pySTAR_FC/contrib/DeepGazeII``` and ```pySTAR_FC/contrib/ICF``` folders respectively (only the checkpoint files (```ckpt.data```, ```ckpt.index``` and ```ckpt.meta```) and ```centerbias.npy``` for each model).

Download SALICONtf from [https://github.com/ykotseruba/SALICONtf] and place the files it in ```pySTAR-FC/contrib/SALICONtf```. Download pre-trained SALICONtf weights:
```
cd contrib/SALICONtf/models
sh download_pretrained_weights.sh
sh download_vgg_weights.sh
```

## Running STAR-FC

Below are instructions on how to run a demo of STAR-FC on a single image (```Yarbus_scaled.jpg``` in ```images``` folder).

If STAR-FC was build using the recommended Dockerfile, use the following command:
```
sh docker_scripts/run.sh -v -c config_files/test.ini
```

Without Docker use the following command:
```
python3 src/STAR_FC.py -v -c config_files/test.ini
```

There are only two command line options available:
* -v for visualization (optional)
* -c for config file in .ini format

All internal parameters of the algorihtm are set via configuration file (for available options and purpose of each parameter see example config file `config_files/template_config.ini`).

Should you have any questions, feel free to raise an issue or email yulia_k@eecs.yorku.ca.

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
