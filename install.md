# Installation


## Clone Repository

```
git clone https://github.com/facebookresearch/holotorch
```

## Clone submodules (external packages not on pip)

We're using a few packages that are not on pip/conda, but are useful to have.
We include them as submodules and one needs to pull them from the corresponding git repos as follows:

```
git submodule update --init --recursive
```

## Create a new Conda Environment
Optional: Create a new conda environment. We used python 3.8 due to compability issues with some of the hardware drivers (Cameras etc.), but feel free to use a newer version.

```
conda create -n holotorch python=3.8
```

## How to install packages?

We recommend using only "pip" commands instead of "conda" commands to install packages since we've experienced problems when we mixed commands. However, any method of installing packages will work. Here we describe the way it works for us:

1. Install PyTorch (we do this before installing the other requirements since this depends on your local hardware configuration)

Insrtuctions found here: https://pytorch.org/get-started/locally/

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

2. Install requirements from our "requirements.txt"

Cd into the main folder and run:

```
pip install -r requirements.txt
```

## Download Div2k for an example dataset

This not necessary, but if you want to download a large image dataset, we recommend Div2k, which you can download div2k from here:
https://data.vision.ee.ethz.ch/cvl/DIV2K/