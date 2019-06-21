# -*- coding:utf-8 -*-

import torch

__version__ = '0.3.0'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
