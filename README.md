# GraB-and-Go

## Quick Access

- [wandb project](https://wandb.ai/grab/grab-cifar10)
- [TestPyPI package](https://test.pypi.org/project/grabngo/)

### Documents

- [Overleaf](https://www.overleaf.com/project/646ad5622b22b94347c78d6a)

Project based on [EugeneLYC/GraB](https://github.com/EugeneLYC/GraB)

## Development Environment Setup

First, install the conda environment

```shell
# To firstly install the environment
conda env create -f environment.yml
```

Then, install the package in editable mode.

```shell
pip install -e .
```

## Build and Release to PyPi

```shell
make build
```
