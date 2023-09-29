import torch
from glcic.networks.discriminators import *
from glcic.networks.completion_network import CompletionNetwork
from glcic.utils import (
    update_replacement_val,
    generate_mask,
    apply_mask,
    postprocess,
)


def train_discriminator(
    discriminator: Discriminator,
    cn: CompletionNetwork,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    num_batch: int,
    replacement_val: torch.tensor,
    info=True,
) -> list:
    """
    This function trains the discriminator network for num_batch batches.
    It can resume its training from a checkpoint.
    """
    # set up
    loss_list = []
    batch_size = dataloader.batch_size
    iterator = iter(dataloader)
    is_cuda = next(discriminator.parameters()).is_cuda
    if is_cuda:
        replacement_val = replacement_val.cuda()
    compute_loss = torch.nn.BCELoss()
    # make sure that optimizer corresponds to discriminator !

    cn.eval()
    discriminator.train()

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
        mask_localizations, mask = generate_mask(batch_size, is_cuda=is_cuda)
        masked_batch = apply_mask(initial_batch, mask, replacement_val)

        # forward + backward (initial images)
        preds = discriminator(initial_batch, mask_localizations)
        loss = compute_loss(preds, torch.ones_like(preds))
        l1 = float(loss)

        # forward + backward (completed images)
        with torch.no_grad():
            masked_batch = cn.forward(
                torch.cat((masked_batch, mask[:, None, :, :]), dim=1)
            ).detach()
            postprocessed = postprocess(masked_batch, mask, mask_localizations)
        preds = discriminator(masked_batch, mask_localizations)
        loss += compute_loss(preds, torch.zeros_like(preds))
        loss.backward()
        l2 = float(loss)

        optimizer.step()

        # saving loss
        loss_list.append((l1 + l2) / 2)
        if info:
            print(f"loss: {l1} for non-completed images, {l2} for completed images")
    return loss_list
