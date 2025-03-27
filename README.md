# PFITRE

## Overview
This repository contains Perception Fused Iterative Tomography Reconstruction Engine (PFITRE), which integrates a convolutional neural network (CNN) with perceptional knowledge as a smart regularizer into an iterative solving engine to correct limited-angle induced artifacts in X-ray tomography images. We demonstrate the effectiveness of the proposed approach using various experimental datasets obtained with different x-ray microscopy techniques. 

The model weights can be downloaded using the following link:
https://drive.google.com/file/d/1rqop4dAZ5QSjZluPkQnnMj5Qkmn5gtKo/view?usp=drive_link


## Examples

We provide an approach for using PFITRE for iterative correction, as well as a one-time post-correction option with the pretrained network for testing and comparison.

[PFITRE for 2D and 3D tomography images](https://github.com/chonghangzhao/PFITRE/blob/main/Demo/PFITRE_Demo_Colab.ipynb)


## Citing Us
If you use PFITRE model, we would appreciate your references to [our paper](https://arxiv.org/abs/2503.19248).

```
@article{PFITRE,
  title = {Limited-angle x-ray nano-tomography with machine-learning enabled iterative reconstruction engine},
  author = {Chonghang Zhao, Mingyuan Ge, Xiaogang Yang, Yong S. Chu, Hanfei Yan},
  Eprint = {10.48550/arXiv.2503.19248},
  year = {2025}
}
```
