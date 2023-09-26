import torch
from glcic.networks.completion_network import CompletionNetwork
from glcic.utils import (
    generate_mask,
    apply_mask,
    compute_mse_loss,
    update_replacement_val,
)


def train_cn(
    cn: CompletionNetwork,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    num_batch: int,
    replacement_val: torch.tensor,
    info=True,
) -> list:
    """
    This function trains the CN network for num_batch batches.
    It can resume its training from a checkpoint.

    Args:
        cn (CompletionNetwork): the completion network to train
        optimizer (torch.optim.Optimizer): the optimizer to use
        dataloader (torch.utils.data.DataLoader): the dataloader to use
        num_batch (int): the number of batch to train
        replacement_val (torch.tensor): the replacement value to use for the masked regions
        info (bool, optional): whether to print info. Defaults to True.

    Returns:
        list: the list of the losses during the training session
    """
    # set up
    loss_list = []
    batch_size = dataloader.batch_size
    iterator = iter(dataloader)
    is_cuda = next(cn.parameters()).is_cuda
    if is_cuda:
        replacement_val = replacement_val.cuda()

    # batch iterations
    for i in range(num_batch):
        if info:
            print(f"\n### BATCH {i+1}/{num_batch} ###")  # to be replaced by tqdm ?
        optimizer.zero_grad()

        # load batch
        initial_batch = next(iterator)[0]
        if is_cuda:
            initial_batch = initial_batch.cuda()
        update_replacement_val(replacement_val, initial_batch)

        # mask
        _, mask = generate_mask(batch_size, is_cuda=is_cuda)
        batch = apply_mask(initial_batch, mask, replacement_val)

        # forward + backward + optimize
        batch = cn.forward(batch)
        loss = compute_mse_loss(initial_batch, batch, mask)
        loss.backward()
        optimizer.step()

        # saving loss
        loss_list.append(float(loss))
        if info:
            print(f"loss: {float(loss)}")
    return loss_list
