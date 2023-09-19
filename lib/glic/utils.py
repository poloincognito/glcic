import numpy as np
import datetime
import os
import pickle

import torch
import torchvision

from glic.networks.completion_network import CompletionNetwork


def generate_mask(batch_size: int, is_cuda=False) -> tuple:
    """
    Generate a random boolean mask for the CN training.
    It first generates a 128*128 mask, that will be used by the context discriminator,
    and then generates a submask whose height and width are randomly selected from [96, 128].
    """
    local_mask = []
    erase_mask = torch.zeros((batch_size, 256, 256), dtype=torch.bool)
    for i in range(batch_size):
        # local mask
        w0, h0 = np.random.randint(0, 128), np.random.randint(0, 128)
        local_mask.append([h0, h0 + 128, w0, w0 + 128])
        # random submask
        w, h = np.random.randint(96, 128), np.random.randint(96, 128)
        w1, h1 = np.random.randint(0, 128 - w), np.random.randint(0, 128 - h)
        erase_mask[i, h0 + h1 : h0 + h, w0 + w1 : w0 + w] = True
    if is_cuda:
        erase_mask = erase_mask.cuda()
    return local_mask, erase_mask


def apply_mask(
    batch: torch.tensor, erase_mask: torch.tensor, replacement_val: torch.tensor
) -> torch.tensor:
    """
    Apply the boolean mask to the batch, replacing all corrsponding pixels with replacement_val.
    """
    masked = (
        batch * (~erase_mask[:, None, :, :])
        + replacement_val[None, :, None, None] * erase_mask[:, None, :, :]
    )
    return masked


@torch.jit.script
def compute_mse_loss(output, target, erase_mask):
    """Compute the MSE loss between the output and the target,
    only considering the masked region."""
    residual_matrix = (erase_mask[:, None, :, :] * (output - target)) ** 2
    mse_batch = torch.sum(residual_matrix, (-1, -2, -3))
    return torch.mean(mse_batch)


def get_current_datetime_string():
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute

    # Format the date and time components into a string
    date_time_string = f"{year}{month:02d}{day:02d}{hour:02d}{minute:02d}"

    return date_time_string


def get_latest_file_from_dir(dir: str) -> str:
    """
    Get the latest file from a directory.
    """
    files = os.listdir(dir)
    paths = [os.path.join(dir, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def list_files(dir: str):
    """
    This function lists all the files in a directory and its subdirectories.
    """
    files_list = []
    for path, subdirs, files in os.walk(dir):
        for name in files:
            full_path = path + "/" + name
            files_list.append(full_path)
    return files_list


class CustomSampler(torch.utils.data.Sampler):
    """Samples elements sequentially, always in the same order, starting from start_idx.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, start_idx) -> None:
        self.start_idx = start_idx
        self.total_len = len(data_source)

    def __iter__(self):
        return iter(range(self.start_idx, self.total_len))

    def __len__(self) -> int:
        return self.total_len - self.start_idx


def get_dataloader(data_dir: str, resume_path: str, batch_size):
    resume_index = list_files(data_dir).index(resume_path)
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_dir, transform=torchvision.transforms.ToTensor()
    )
    custom_sampler = CustomSampler(train_dataset, resume_index)
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=custom_sampler
    )
    return dataloader


def save_checkpoint(
    dir: str,
    cn: CompletionNetwork,
    optimizer: torch.optim.Optimizer,
    loss: list,
    batch: int,
    resume_path: str,
    replacement_val: torch.tensor,
):
    """
    This function saves the current state of the training.

    Args:
        dir (str): directory where to save the checkpoint
        cn (CompletionNetwork): the completion network whose state will be saved
        optimizer (torch.optim.Optimizer): the optimizer whose state will be saved
        loss (list): the current loss list (dimension 2), containing the loss list of each session
        batch (int): the current batch index
        resume_path (str): the path of the image from which the training will resume
    """
    torch.save(
        {
            "cn": cn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
            "batch": batch,
            "resume_path": resume_path,
            "replacement_val": replacement_val,
        },
        dir + "/" + get_current_datetime_string(),
    )


def load_checkpoint(
    dir: str,
    cn: CompletionNetwork,
    optimizer: torch.optim.Optimizer,
):
    """
    This function loads the latest checkpoint from a directory.
    It returns the loss list, the number of batch since the training began,
    and the path of the image from which the training will resume.
    """
    checkpoint = torch.load(get_latest_file_from_dir(dir))
    cn.load_state_dict(checkpoint["cn"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        checkpoint["loss"],
        checkpoint["batch"],
        checkpoint["resume_path"],
        checkpoint["replacement_val"],
    )


def update_resume_path(
    datadir: str, resume_path: str, batchnum: int, batchsize: int
) -> str:
    """
    This function returns the path of the image from which the training will resume.
    """
    files_list = list_files(datadir)
    resume_index = files_list.index(resume_path)
    new_resume_path = files_list[resume_index + batchnum * batchsize]
    return new_resume_path


def apply_local_parameters(batch, local_parameters):
    """
    This function extract the local images defined by the local_parameters,
    and returns them as a batch.

    Eg: local_parameters = [[h1,h2,w1,w2], ...]
    The returned tensor results will look like results[0] = batch[0,:,h1:h2,w1:w2]
    """
    slices = []
    for idx, coords in enumerate(local_parameters):
        h1, h2, w1, w2 = coords
        # Extract slices from the batch using the coordinates
        slice_tensor = batch[idx, :, h1:h2, w1:w2]
        # Append the extracted slice to the list
        slices.append(slice_tensor)
    return torch.stack(slices, dim=0)
