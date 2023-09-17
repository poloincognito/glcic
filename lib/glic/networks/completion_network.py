import pandas as pd
import torch.nn as nn
from importlib import resources as impresources

import glic
from glic.networks.net_tools import build_layer


class CompletionNetwork(nn.Module):

    """
    This class implements the Completion Network (CN) as described in the paper.
    """

    def __init__(self):
        super().__init__()
        layers = []
        cn_params_file = str(impresources.files(glic)) + "/networks/cn_layers_params.p"
        cn_layers_params = pd.read_pickle(
            cn_params_file
        )  # load the parameters of the CN layers
        for i, row in cn_layers_params.iterrows():  # iteratively builds the CN layers
            layers.append(build_layer(*row.values))
        self.cn_net = nn.Sequential(*layers)  # assembles the CN layers

    def forward(self, X):
        vec = self.cn_net(X)
        return vec
