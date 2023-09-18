import numpy as np
import torch


def generate_mask(batch_size: int) -> tuple:
    """
    Generate a random boolean mask for the CN training.
    It first generates a 128*128 mask, that will be used by the context discriminator,
    and then generates a submask whose height and width are randomly selected from [96, 128].
    """
    local_mask = torch.zeros((batch_size, 256, 256), dtype=torch.bool)
    erase_mask = torch.zeros((batch_size, 256, 256), dtype=torch.bool)
    for i in range(batch_size):
        # local mask
        w, h = np.random.randint(0, 128), np.random.randint(0, 128)
        local_mask[i, h : h + 128, w : w + 128] = True
        # random submask
        subw, subh = np.random.randint(0, 32), np.random.randint(0, 32)
        local_mask[i, h + subh : h + 128, w + subw : w + 128] = True
    return local_mask, erase_mask


def apply_mask(batch, erase_mask, replacement_val):
    return (
        batch * (~erase_mask[:, None, :, :])
        + replacement_val[None, :, None, None] * erase_mask[:, None, :, :]
    )
