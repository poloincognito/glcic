import argparse

import torch

from glic.training.cn_training import train_cn
from glic.networks.completion_network import CompletionNetwork
from glic.utils import (
    load_checkpoint,
    save_checkpoint,
    get_dataloader,
    update_resume_path,
    list_files,
)

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", default=96, help="size of the batches", type=int)
parser.add_argument("--batchnum", default=100, help="number of batch to run", type=int)
parser.add_argument(
    "--datadir", default="../data/train/", help="directory of the data", type=str
)
parser.add_argument(
    "--checkpointsdir",
    default="../logs/checkpoints/",
    help="directory of the checkpoints",
    type=str,
)


def main(args):
    # loads the latest checkpoint
    cn = CompletionNetwork()
    optimizer = torch.optim.Adadelta(cn.parameters(), lr=2e-4)
    loss_list, batch, resume_path, replacement_val = load_checkpoint(
        args.checkpointsdir, cn, optimizer
    )
    dataloader = get_dataloader(args.datadir, resume_path, batch_size=args.batchsize)
    train_sessions = len(dataloader) // args.batchsize

    # info
    print("The dataset is located at ", args.datadir)
    print(len(list_files(args.datadir)), " is the number of files")
    resume_index = list_files(args.datadir).index(resume_path)
    print(resume_index, " is the resume index")
    print(len(dataloader), " is the length of the dataloader")
    print(f"{train_sessions} train sessions planned")

    for session in range(train_sessions):  # train_sessions
        # trains the completion network
        current_loss_list = train_cn(
            cn, optimizer, dataloader, args.batchnum, replacement_val, info=False
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
