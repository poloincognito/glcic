# Presentation

This project is based on the content of a computer vision course. I try to re-implement the following paper from 2017:
*Globally and Locally Consistent Image Completion*, http://dx.doi.org/10.1145/3072959.3073659

The project is developed using Git.
The implementation uses `torch` and CUDA for GPU training.
I use an ssh connection to a university machine with an NVIDIA GeForce RTX 3090 for training.

## Install

I created a python library called `glic` (*Globally and Locally consistent Image Completion*) to achieve this challenge.
One can install it with: `pip install -e .` from the root directory of the library (`-e` is useful if you aim to keep modifying the library).

## Discover

The best way to discover the library is to go through the notebooks.
Deep dive in the code for further documentation.

# *Completion Network*

I first built the image completion network, see ```lib/glic/networks/completion_network.py```.

The initial training can be remotely launched using ```nohup python tain_cn.py```.

![image info](./figures/cn_training.png)