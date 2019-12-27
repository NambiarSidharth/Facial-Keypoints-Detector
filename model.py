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
        self.conv1 = nn.Conv2d(1,32,5) #image size will be (32,220,220)
        self.conv2 = nn.Conv2d(32,64,5)
        self.fc1 = nn.Linear(64*53*53,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,136)
        self.drop1 = nn.Dropout(p=0.4)
        
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = self.drop1(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = self.fc3(x)
        return x
        