import torch
import torch.nn as nn
import torch.nn.functional as F

class PathConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin(x))