import argparse

import torch

from glic.training.cn_training import train_cn
from glic.networks.completion_network import CompletionNetwork
from glic.utils import (
    load_checkpoint,
    save_checkpoint,
    get_dataloader,
    update_resume_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("batchsize", help="size of the batches", type=int)
parser.add_argument("batchnum", help="number of batch to run", type=int)
parser.add_argument("datadir", help="directory of the data", type=str)
parser.add_argument("checkpointsdir", help="directory of the checkpoints", type=str)
parser.add_argument("-info", type=int, default=1)


def main(args):
    # loads the latest checkpoint
    cn = CompletionNetwork()
    optimizer = torch.optim.Adadelta(cn.parameters(), lr=2e-4)
    loss_list, batch, resume_path, replacement_val = load_checkpoint(
        args.checkpointsdir, cn, optimizer
    )
    dataloader = get_dataloader(args.datadir, resume_path, batch_size=args.batchsize)

    # trains the completion network
    current_loss_list = train_cn(
        cn, optimizer, dataloader, args.batchnum, replacement_val, info=args.info
    )

    # saves the checkpoint
    resume_path = update_resume_path(
        args.datadir, resume_path, args.batchnum, args.batchsize
    )
    batch += args.batchnum
    loss_list += [current_loss_list]
    save_checkpoint(
        args.checkpointsdir,
        cn,
        optimizer,
        loss_list,
        batch + args.batchnum,
        resume_path,
        replacement_val,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
