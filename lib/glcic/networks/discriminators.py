import pandas as pd
import torch
import torch.nn as nn
from importlib import resources as impresources

import glcic
from glcic.networks.net_tools import build_layer
from glcic.utils import apply_local_parameters


class GlobalDiscriminator(nn.Module):

    """
    This class implements the global discriminator as described in the paper.
    Its architecure is defined by gd_layers_params.p
    """

    def __init__(self):
        super().__init__()
        layers = []
        gd_params_file = str(impresources.files(glcic)) + "/networks/gd_layers_params.p"
        gd_layers_params = pd.read_pickle(
            gd_params_file
        )  # load the parameters of the GD layers
        for i, row in gd_layers_params.iterrows():  # iteratively builds the GD layers
            layers.append(build_layer(*row.values))
        # last FC layer
        layers.append(nn.Flatten(start_dim=1))
        default_size = 8192
        layers.append(nn.Linear(default_size, 1024))
        layers.append(nn.ReLU())
        self.gd_net = nn.Sequential(*layers)  # assembles the GD layers

    def forward(self, X):
        vec = self.gd_net(X)
        return vec

    # save and load
    def save(self, path):
        torch.save(self.state_dict(), path)
        print("Save: state_dict saved in {}".format(path))

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print("Load: load_state dict from {}".format(path))


class LocalDiscriminator(nn.Module):

    """
    This class implements the local discriminator as described in the paper.
    Its architecure is defined by ld_layers_params.p
    """

    def __init__(self):
        super().__init__()
        layers = []
        ld_params_file = str(impresources.files(glcic)) + "/networks/ld_layers_params.p"
        ld_layers_params = pd.read_pickle(
            ld_params_file
        )  # load the parameters of the LD layers
        for i, row in ld_layers_params.iterrows():  # iteratively builds the LD layers
            layers.append(build_layer(*row.values))
        # last FC layer
        layers.append(nn.Flatten(start_dim=1))
        default_size = 8192
        layers.append(nn.Linear(default_size, 1024))
        layers.append(nn.ReLU())
        self.ld_net = nn.Sequential(*layers)  # assembles the LD layers

    def forward(self, X):
        vec = self.ld_net(X)
        return vec

    # save and load
    def save(self, path):
        torch.save(self.state_dict(), path)
        print("Save: state_dict saved in {}".format(path))

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print("Load: load_state dict from {}".format(path))


class Discriminator(nn.Module):
    """
    This class assembles the global and local discriminators.
    """

    def __init__(self):
        super().__init__()
        self.global_discriminator = GlobalDiscriminator()
        self.local_discriminator = LocalDiscriminator()
        self.last_layer = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())

    def forward(self, x, local_parameters):
        local_x = apply_local_parameters(x, local_parameters)
        output1 = self.global_discriminator(x)
        output2 = self.local_discriminator(local_x)
        assert output1.ndim > 1, "The discriminator only accepts batch."
        prediction = self.last_layer(torch.cat((output1, output2), dim=1))
        return prediction
