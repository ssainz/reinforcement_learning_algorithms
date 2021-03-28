import torch
import torch.nn as nn
import torch.nn.functional as F


class NetConf(nn.Module):
    def __init__(self, net_config):
        sizes = net_config["layers"]
        if net_config["non_linear_function"] == "relu":
            self.nonlinear = F.relu
        elif net_config["non_linear_function"] == "tanh":
            self.nonlinear = F.tanh
        super(NetConf, self).__init__()
        self.input_size = sizes[0]
        self.out_size = sizes[-1]
        self.output_shape = net_config["output_shape"] # E.g. [4,10], it means 4 is first dimension, 10 is second dimension.
        self.layers = nn.ModuleList()
        for i in range(len(sizes)-1):
            fc1 = nn.Linear(sizes[i], sizes[i+1])
            self.layers.append(fc1)
    def forward(self, x_orig):
        #x = x.view(-1, self.input_size)
        x = x_orig # for Maze: (16,), for SAV: (1 , 42)
        for layer in self.layers:
            x = layer(x)
            x = self.nonlinear(x)
        x = x.reshape(self.output_shape)
        if len(x.size()) == 1:
            x = F.softmax(x,dim=0) # dim is the dimension to destroy. Only 0 dimension here.
                                   # Left with if condition for clarity.
        else:
            x = F.softmax(x, dim=( len(x.size()) - 1) ) # We want to destroy the last dimension when more one dimensions are present.
                                                        # x.size() returns [2,3] for example, len([2,3]) returns 2 due to two elements.
                                                        # -1 is to adjust for the 0-index dimensions. We pick last dimension.
        return x
    def inits(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)