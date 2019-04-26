import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

## check your python version and pytorch version
## it would be great if we get consistent and use python 3.6x and pytorch 0.4.1
import sys
print(sys.version)
print(torch)
print(torch.__version__)

## test pytorch 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

class MLP(nn.Module):
    ## a very simple MLP model
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

mlp = MLP(20,10)

x = torch.randn((5, 20))
y = torch.Tensor([0,3,5,2,3]).long().reshape(-1)

pred = mlp(x)
print(y.shape)

criterion = nn.CrossEntropyLoss()

loss = criterion(pred, y)

## if you can run all cells without error likely your pytorch is working
print(loss)

print("we made it woot")