import torch
import torchvision

import argparse

from glic.utils import *
from glic.networks.completion_network import CompletionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("numbatch", help="number of batch to run", type=int)
parser.add_argument("logdir", help="directory for the logs", type=str)


def main(num_batch: int, log_dir: str, data_dir: str = "data/"):
    # load
    log, cn = load_cn_training_log(log_dir)
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_dir, transform=torchvision.transforms.ToTensor()
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.numbatch, args.logdir)
