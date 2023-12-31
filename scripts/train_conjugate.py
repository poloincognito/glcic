import argparse

import torch

from glcic.trainers.conjugate_trainer import train
from glcic.networks.discriminators import Discriminator
from glcic.networks.completion_network import CompletionNetwork
from glcic.utils import (
    load_conjugate_checkpoint,
    save_conjugate_checkpoint,
    get_dataloader,
    update_resume_path,
    list_files,
)

"""
This script trains the discriminator on the training set.

Optional arguments are:
    --batchsize: size of the batches
    --batchnum: number of batch to run between each checkpoints
    --datadir: directory of the training data
    --checkpointsdir: directory of the checkpoints
    --cndir: path of the completion network

It then runs the traing until the training set is exhausted.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", default=48, help="size of the batches", type=int)
parser.add_argument("--batchnum", default=200, help="number of batch to run", type=int)
parser.add_argument(
    "--datadir", default="../data/train/", help="directory of the data", type=str
)
parser.add_argument(
    "--checkpointsdir",
    default="../logs/phase3_checkpoints",
    help="directory of the checkpoints",
    type=str,
)
parser.add_argument(
    "--cndir",
    default="../logs/models/cn_scrapped_weights",
    help="directory of the trained completion network",
    type=str,
)
parser.add_argument(
    "--discriminatordir",
    default="../logs/models/trained_discriminator",
    help="directory of the trained completion network",
    type=str,
)
parser.add_argument(
    "--trainsessions", default=-1, help="number of training sessions", type=int
)


def main(args):
    # load models
    cn = CompletionNetwork()
    cn.load(args.cndir)
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(args.discriminatordir))

    # optimizers
    cn_optimizer = torch.optim.Adadelta(cn.parameters())
    discriminator_optimizer = torch.optim.Adadelta(discriminator.parameters())

    # load checkpoints
    loss_list, batch, resume_path, replacement_val = load_conjugate_checkpoint(
        args.checkpointsdir,
        [cn, discriminator],
        [cn_optimizer, discriminator_optimizer],
    )
    dataloader = get_dataloader(args.datadir, resume_path, batch_size=args.batchsize)
    train_sessions = len(dataloader) // args.batchsize

    # info
    print("The cn model is located at ", args.cndir)
    print("The discriminator model is located at ", args.discriminatordir)
    print("The dataset is located at ", args.datadir)
    print(len(list_files(args.datadir)), " is the number of files")
    resume_index = list_files(args.datadir).index(resume_path)
    print(resume_index, " is the resume index")
    print(len(dataloader), " is the length of the dataloader")
    if args.trainsessions >= 0:
        train_sessions = min(args.trainsessions, train_sessions)
    print(f"{train_sessions} train sessions planned")

    for session in range(train_sessions):  # train_sessions
        # trains the completion network
        current_loss_list = train(
            cn,
            discriminator,
            cn_optimizer,
            discriminator_optimizer,
            dataloader,
            args.batchnum,
            replacement_val,
            info=False,
        )

        # saves the checkpoint
        resume_path = update_resume_path(
            args.datadir, resume_path, args.batchnum, args.batchsize
        )
        batch += args.batchnum
        loss_list += [current_loss_list]
        save_conjugate_checkpoint(
            args.checkpointsdir,
            [cn, discriminator],
            [cn_optimizer, discriminator_optimizer],
            loss_list,
            batch + args.batchnum,
            resume_path,
            replacement_val,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
