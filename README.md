# TensorFlow tutorials

## TensorFlow in Ubuntu 22.04

(Instructions extracted from https://www.tensorflow.org/install/pip#step-by-step_instructions)

1. Install Miniconda

```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

2. Create conda environment

```
conda create --name tf python=3.9
conda activate tf
```

3. Install Nvidia GPU driver

```
sudo apt install nvidia-utils-525
```

3.1 Verify installation

```
nvidia-smi
```

4. Install CUDA and cuDNN with conda

```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

5. Configure system paths

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

Alternatively, if that variable does not have a value, you can execute this command instead:
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/
```

5.1 (Optional) To automatically enable it every time this conda environment is activated

```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

6. Install TensorFlow with pip

```
pip install --upgrade pip
pip install tensorflow
```

6.1 Verify installation

```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Troubleshooting

`
Can’t find libdevice directory ${CUDA_DIR}/nvvm/libdevice
`
1. `conda install -c nvidia cuda-nvcc`
2. `sudo ln -s $LD_LIBRARY_PATH /usr/local/cuda-11.2`

## Tensorflow in WSL2

Install CUDA (instructions from https://docs.nvidia.com/cuda/wsl-user-guide/index.html):

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-wsl-ubuntu-12-0-local_12.0.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-0-local_12.0.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```