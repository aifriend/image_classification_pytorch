## ENVIRONMENT SETUP
    pip install --upgrade pip

## PYTORCH INSTALLATION FOR CUDA 11

### Windows
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

### Linux
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

## GPU

### Windows
    Version             Python version	Compiler	Build tools     cuDNN       CUDA
    tensorflow-2.4.0    3.6-3.8	        GCC 7.3.1	Bazel 3.1.0     8.0         11.0

## SUPPORT INSTALLATION
    pip install tf-nightly

## GAN TUTORIAL
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

## TRANSFORMER INSTALL FROM SOURCE

### v1.4.4
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install -e .
