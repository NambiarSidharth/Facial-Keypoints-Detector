# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 23:26:17 2019

@author: Surendran Nambiar
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,)