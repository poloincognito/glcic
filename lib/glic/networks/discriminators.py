import pandas as pd
import torch.nn as nn
from importlib import resources as impresources

import glic
from glic.networks.net_tools import build_layer


class GlobalDiscriminator(nn.Module):

    """
    This class implements the global discriminator as described in the paper.
    """

    def __init__(self):
        super().__init__()
        layers = []
        gd_params_file = str(impresources.files(glic)) + "/networks/gd_layers_params.p"
        gd_layers_params = pd.read_pickle(
            gd_params_file
        )  # load the parameters of the GD layers
        for i, row in gd_layers_params.iterrows():  # iteratively builds the GD layers
            layers.append(build_layer(*row.values))
        # last FC layer
        layers.append(nn.Flatten())
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
    """

    def __init__(self):
        super().__init__()
        layers = []
        ld_params_file = str(impresources.files(glic)) + "/networks/ld_layers_params.p"
        ld_layers_params = pd.read_pickle(
            ld_params_file
        )  # load the parameters of the LD layers
        for i, row in ld_layers_params.iterrows():  # iteratively builds the LD layers
            layers.append(build_layer(*row.values))
        # last FC layer
        layers.append(nn.Flatten())
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
