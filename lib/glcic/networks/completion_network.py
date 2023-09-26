import pandas as pd
import torch
import torch.nn as nn
from importlib import resources as impresources

import glcic
from glcic.networks.net_tools import build_layer


class CompletionNetwork(nn.Module):

    """
    This class implements the Completion Network (CN) as described in the paper.
    """

    def __init__(self):
        super().__init__()
        layers = []
        cn_params_file = str(impresources.files(glcic)) + "/networks/cn_layers_params.p"
        cn_layers_params = pd.read_pickle(
            cn_params_file
        )  # load the parameters of the CN layers
        for i, row in cn_layers_params.iterrows():  # iteratively builds the CN layers
            layers.append(build_layer(*row.values))
        self.cn_net = nn.Sequential(*layers)  # assembles the CN layers

    def forward(self, X):
        vec = self.cn_net(X)
        return vec

    # save and load
    def save(self, path):
        torch.save(self.state_dict(), path)
        print("Save: state_dict saved in {}".format(path))

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print("Load: load_state dict from {}".format(path))
