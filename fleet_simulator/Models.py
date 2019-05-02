import torch
import torchvision
import random
import torch.nn as nn
import torch
import torch.nn.functional as F


class pi_net(nn.Module):
    def __init__(self):
        super(pi_net, self).__init__()
        bias_on = True
        self.linear1 = nn.Linear(36, 64, bias=bias_on)
        self.linear2 = nn.Linear(64, 64, bias=bias_on)
        self.linear3 = nn.Linear(64, 36, bias=bias_on)
        #torch.nn.init.xavier_uniform_(self.linear1)
        #torch.nn.init.xavier_uniform_(self.linear2)

    def forward(self, x):


        # --- 0000 ---- 0000 >>>  z-score normalization
        x = self.linear1(x)
        # print("AFTER linear1 = = = = = = = = = =")
        # print(x)
        # print("AFTER linear1 = = = = = = = = = =")

        x_avg = torch.sum(x)  / 20
        # print("AVG " + str(x_avg) )
        # print("x - x_avg ~~~~~~~~~~~~~~")
        x_minus_x_avg = x - x_avg
        # print(x_minus_x_avg)
        # print("x - x_avg ~~~~~~~~~~~~~~")
        x_std = torch.sum(torch.pow(x_minus_x_avg, 2)) / 20
        # print("VAR " + str(x_std))
        epsilon = 0.0000001
        # print("STD " + str(torch.sqrt(x_std)))
        x_norm = (x_minus_x_avg) / (torch.sqrt(x_std) + epsilon)
        # print("BEFORE sigmoid = = = = = = = = = =")
        # print(x_norm)
        # print("BEFORE sigmoid = = = = = = = = = =")
        #x = F.sigmoid(x_norm)
        x = F.tanh(x_norm)

        x = self.linear2(x)

        x_avg = torch.sum(x) / 40
        x_minus_x_avg = x - x_avg
        x_std = torch.sum(torch.pow(x_minus_x_avg, 2)) / 40
        x_norm = (x_minus_x_avg) / (torch.sqrt(x_std) + epsilon)
        x = F.tanh(x_norm)

        # print("AFTER sigmoid = = = = = = = = = =")
        # print(x)
        # print("AFTER sigmoid = = = = = = = = = =")
        x = self.linear3(x)
        return x.view(-1, 36)


        # --- 0000 ---- 0000 >>>  feature scaling
        # x = self.linear1(x)
        # print("AFTER linear1 = = = = = = = = = =")
        # print(x)
        # print("AFTER linear1 = = = = = = = = = =")

        # x_max = torch.max(x)
        # x_min = torch.min(x)
        # epsilon = 0.00001
        # x_norm = ((x - x_min) / (x_max - x_min + epsilon))

        # print("BEFORE sigmoid = = = = = = = = = =")
        # print(x_norm)
        # print("BEFORE sigmoid = = = = = = = = = =")
        # x = F.sigmoid(x_norm)
        # print("AFTER sigmoid = = = = = = = = = =")
        # print(x)
        # print("AFTER sigmoid = = = = = = = = = =")
        # x = self.linear2(x)
        # return x.view(-1, 4)
