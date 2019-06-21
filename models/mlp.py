# -*- coding:utf-8 -*-

import torch.nn as nn


class LinearNet(nn.Module):
    """LinearNet: Simple MLP with activation funcition 'ReLU'"""

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(LinearNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
