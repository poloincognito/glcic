import torch
from glcic.networks.discriminators import *
from glcic.networks.completion_network import CompletionNetwork
from glcic.utils import (
    update_replacement_val,
    generate_mask,
    apply_mask,
    postprocess,
    compute_mse_loss,
    get_model_grad_norm,
    generate_random_masks_localization,
    update_moving_average,
    get_grad_norm,
    manually_update_grad,
)


def train(
    cn: CompletionNetwork,
    discriminator: Discriminator,
    cn_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    num_batch: int,
    replacement_val: torch.tensor,
    info=True,
    adaptative_alpha=False,
) -> list:
    """
    This function implements the conjugate training of the completion network and the discriminator.
    Check the README.md for explanatory schemes.
    It can resume its training from a checkpoint.
    """

    # set up
    loss_list = []
    batch_size = dataloader.batch_size
    iterator = iter(dataloader)
    is_cuda = next(discriminator.parameters()).is_cuda
    if is_cuda:
        replacement_val = replacement_val.cuda()
    mse_loss = compute_mse_loss
    bce_loss = torch.nn.BCELoss()
    alpha = 4e-4
    # make sure that the cn_optimizer corresponds to cn !
    # make sure that the discriminator_optimizer corresponds to discriminator !

    cn.train()
    discriminator.train()

    # batch iterations
    for i in range(num_batch):
        if info:
            print(f"\n### BATCH {i+1}/{num_batch} ###")  # to be replaced by tqdm ?

        # load batch
        initial_batch = next(iterator)[0]
        if is_cuda:
            initial_batch = initial_batch.cuda()
        update_replacement_val(replacement_val, initial_batch)

        ### discriminator ###
        discriminator_optimizer.zero_grad()

        # mask
        mask_localizations, mask = generate_mask(batch_size, is_cuda=is_cuda)
        masked_batch = apply_mask(initial_batch, mask, replacement_val)

        # fake forward
        completed_batch = cn.forward(
            torch.cat((masked_batch, mask[:, None, :, :]), dim=1)
        ).detach()
        completed_batch = postprocess(completed_batch, mask, mask_localizations)
        fake_preds = discriminator(completed_batch, mask_localizations)
        l1 = bce_loss(fake_preds, torch.zeros_like(fake_preds))

        # real forward
        random_mask_localizations = generate_random_masks_localization(batch_size)
        real_preds = discriminator(initial_batch, random_mask_localizations)
        l2 = bce_loss(real_preds, torch.ones_like(real_preds))

        d_loss = alpha * (l1 + l2) / 2

        # backward
        d_loss.backward()

        # update
        discriminator_optimizer.step()

        ### completion network ###
        cn_optimizer.zero_grad()

        # mask
        mask_localizations, mask = generate_mask(batch_size, is_cuda=is_cuda)
        masked_batch = apply_mask(initial_batch, mask, replacement_val)

        # forward
        masked_batch = cn.forward(torch.cat((masked_batch, mask[:, None, :, :]), dim=1))
        l1 = mse_loss(initial_batch, masked_batch, mask)
        preds = discriminator(
            masked_batch, mask_localizations
        )  # no postprocess, backward propagation needed
        l2 = alpha * bce_loss(preds, torch.ones_like(preds))

        # backward, with adaptative alpha to keep mse grad and bce grad in the same range
        if adaptative_alpha:
            grad1 = torch.autograd.grad(l1, cn.parameters(), retain_graph=True)
            grad2 = torch.autograd.grad(l2, cn.parameters())
            grad1_norm, grad2_norm = get_grad_norm(grad1), get_grad_norm(grad2)
            print("mse grad norm: ", grad1_norm, ", bce grad norm: ", grad2_norm)
            alpha = update_moving_average(alpha, grad1_norm / grad2_norm, 0.99)
            print("alpha: ", alpha)
            manually_update_grad(cn.parameters(), grad1, grad2)
        else:
            (l1 + l2).backward()

        # update
        cn_optimizer.step()

        # saving loss
        cn_loss, discriminator_loss = float(l1 + l2), float(d_loss)
        loss_list.append((cn_loss, discriminator_loss))
        if info:
            print(
                f"loss: {cn_loss} for the completion loss, {discriminator_loss} for the discriminator loss"
            )
    return loss_list
