import torch.nn as nn


def build_layer(
    conv_type: str,
    kernel: int,
    dilat: int,
    stri: int,
    inputs: int,
    outputs: int,
    batchnorm: bool,
    activation="ReLU",
) -> nn.Sequential:
    """
    This function creates a pytorch neural network layer with the specified parameters.

    Args:
        conv_type (str): indicates if it is a convolutional or deconvolutional layer.
        kernel (int): kernel size.
        dilat (int): dilation.
        stri (int): stride.
        inputs (int): number of input channels.
        outputs (int): number of output channels.
        batchnorm (bool): using batchnorm.
        activation (str): activation function used.


    Returns:
        layer (nn.Sequential): desired pytorch neural network layer,
        with its batchnorm and activation layers."""

    layers = []

    if conv_type == "conv":
        pad = kernel // 2 * dilat  # conservation of size
        conv_layer = nn.Conv2d(
            inputs,
            outputs,
            kernel,
            dilation=dilat,
            stride=int(stri),
            padding=pad,
            # bias=not (batchnorm),  # batchnorm2d already has bias
        )
        layers.append(conv_layer)
    else:
        eff_stri = int(1 / stri)  # effective stride
        pad = (kernel - eff_stri) // 2  # conservation of size
        deconv_layer = nn.ConvTranspose2d(
            inputs,
            outputs,
            kernel,
            dilation=dilat,
            stride=eff_stri,
            padding=pad,
            # bias=not (batchnorm),  # batchnorm2d already has bias
        )
        layers.append(deconv_layer)

    # batchnorm
    if batchnorm:
        layers.append(nn.BatchNorm2d(outputs))

    # activation layer
    activation = activation.lower()
    if activation == "relu":
        layers.append(nn.ReLU())
    elif activation == "leakyrelu":
        layers.append(nn.LeakyReLU())
    elif activation == "sigmoid":
        layers.append(nn.Sigmoid())

    # assembling everything
    layer = nn.Sequential(*layers)

    return layer
