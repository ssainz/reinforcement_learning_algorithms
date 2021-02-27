import torch
import torch.nn as nn
import torch.nn.functional as F


class NetConf(nn.Module):
    def __init__(self, sizes):
        super(NetConf, self).__init__()
        self.input_size = sizes[0]
        self.out_size = sizes[-1]
        self.layers = nn.ModuleList()
        for i in range(len(sizes)-1):
            fc1 = nn.Linear(sizes[i], sizes[i+1])
            self.layers.append(fc1)
    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = F.softmax(x, dim=1)
        return x.view(-1, self.out_size)
    def inits(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)