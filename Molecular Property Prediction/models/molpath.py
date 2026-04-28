import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention
from models.path_conv import PathConv 

class MolPath(nn.Module):
    def __init__(self, input_dim, hidden_dim, K=6, lambda_param=0.2):
        super().__init__()
        self.K = K
        self.lambda_param = lambda_param
        self.lin_init = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList([PathConv(hidden_dim, hidden_dim) for _ in range(K)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(K)])
        
        # Learnable skip-connection scaling
        self.alpha = nn.Parameter(torch.tensor([0.1]))
        
        self.gate_nn = nn.Linear(hidden_dim, 1)
        self.pool = GlobalAttention(self.gate_nn)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        h0 = F.relu(self.lin_init(data.x.float()))
        x = h0

        for i in range(self.K):
            x_new = self.convs[i](x)
            x_new = self.bns[i](x_new)
            
            # Improved IRDC with Learnable Scaling Alpha
            # This keeps the signal 'alive' for 150+ epochs
            x = (1 - self.alpha) * x + self.alpha * x_new + (self.lambda_param * h0)
            x = F.relu(x)

        g_repr = self.pool(x, data.batch)
        return self.predictor(g_repr)