# GraB-and-Go

This is the following work of the
paper [GraB: Finding Provably Better Data Permutations than Random Reshuffling
](https://arxiv.org/abs/2205.10733).

## GraB-sampler

The [grab-sampler](https://github.com/garywei944/grab-sampler) is a separate PyPI
project aimed to release an user-friendly interface of GraB-sampler with using GraB and
FuncTorch.
For research purpose, this repository simply includes the coding base of the
GraB-sampler.

## Quick Access

- [wandb projects](https://wandb.ai/grab/projects)
- [PyPI package](https://pypi.org/project/grab-sampler/)

### Documents

- [GraB-sampler Technique Report](https://www.overleaf.com/project/646a9aa45e534c915b8d2685)
- [ML Optimization Literature Review](https://www.overleaf.com/read/tgxwgzcgjbpk#c61264)
- [Abandoned Workshop Overleaf](https://www.overleaf.com/project/646ad5622b22b94347c78d6a) [Deprecated since 6/1/2023]

### Previous Code Repositories

- [grab-sampler](https://github.com/garywei944/grab-sampler) 
- [EugeneLYC/GraB](https://github.com/EugeneLYC/GraB) - The original released codebase
  for the paper.

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

## Build and Release to PyPI

```shell
make release
```
