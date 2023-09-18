import numpy as np
import torch


def generate_mask(batch_size: int) -> tuple:
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
    return local_mask, erase_mask


def apply_mask(batch, erase_mask, replacement_val):
    return (
        batch * (~erase_mask[:, None, :, :])
        + replacement_val[None, :, None, None] * erase_mask[:, None, :, :]
    )
