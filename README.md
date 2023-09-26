# Presentation

This project is based on the content of a computer vision course. As a final project, I try to re-implement the following paper from 2017:
*Globally and Locally Consistent Image Completion*, http://dx.doi.org/10.1145/3072959.3073659

![image info](./figures/glcic_paper.PNG)

The project is developed using Git.
The implementation uses `torch` and CUDA for GPU training.
I use an ssh connection to a university machine with an NVIDIA GeForce RTX 3090 for the training sessions.

## Install

I created a python library called `glcic` (*Globally and Locally Consistent Image Completion*) to achieve this challenge.
One can install it with: `pip install -e .` from the root directory of the library (`-e` is useful if you aim to keep modifying the library).

There is currently no `requirements.txt` file.
Two folders need to be manually created: `data/train` and `logs/checkpoints`.

## Discover

The best way to discover the library is to go through the notebooks.
Deep dive in the code for further documentation.

# *Completion Network*

I first built the image completion network, see ```lib/glic/networks/completion_network.py```.

Its trainer is ```lib/glic/trainers/cn_trainer.py```.

The initial training can be remotely launched using ```nohup python train_cn.py``` (*make sure to initialize the `./logs/checkpoints/` and `./data/train/` directories*).

*TO BE UPDATED !*  

I trained the completion network for 19 hours on an NVIDIA GeForce RTX 3090.

![image info](./figures/cn_training.png)

The whole training would take an estimated time of 37 days.

To continue the project, I scrapped some weights from `https://github.com/otenim/GLCIC-PyTorch`.

After the weights transfer, the completion ability of the CN network was:

*TO BE UPDATED ! FIGURES*  

# *Discriminator*

The implementation of the discriminators is in ```lib/glic/networks/discriminators.py```.

Its trainer is ```lib/glic/trainers/discriminators_trainer.py```.

I trained the discriminator for x hours on an NVIDIA GeForce RTX 3090 and obtained:

*TO BE COMPLETED ! FIGURES*