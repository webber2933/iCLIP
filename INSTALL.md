## Installation

**Requirements**

- Python >= 3.7
- [Pytorch](https://pytorch.org/) = 1.7.1
- [yacs](https://github.com/rbgirshick/yacs)
- [OpenCV](https://opencv.org/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm)
- [FFmpeg](https://www.ffmpeg.org/)
- Linux and Nvidia GPUs (we used one 3080 GPU for J-HMDB and UCF101-24)

We recommend to setup the environment with Anaconda, 
the step-by-step installation script is shown below.

```bash
conda create -n iCLIP python=3.7
conda activate iCLIP

# we use cuda version = 11.0
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

conda install av -c conda-forge
conda install cython
pip install einops

git clone https://github.com/webber2933/iCLIP.git
cd iCLIP

# If you encounter errors, you can try the below 2 commands
TORCH_CUDA_ARCH_LIST="8.0" python setup.py install
export TORCH_CUDA_ARCH_LIST=8.0

pip install -e .    # install other dependencies

```
### Errors frequently encountered + solutions

1. If the last command fails with `No CUDA found` error, it might be because Anaconda installed the CPU version of pytorch. Please use `pip install` with your prefered pytorch and cuda versions and try again. For instance, with torch 1.10.0 and cuda 10.2, you would run: `pip install torch==1.10.0+cu102 torchvision --extra-index-url https://download.pytorch.org/whl/cu102`

2. If the error is `fatal error: THC/THC.h: No such file or directory`, I suggest keep your pytorch version between 1.4.0 and 1.10.0, inclusive.

3. Any other issues? Please open an issue.
