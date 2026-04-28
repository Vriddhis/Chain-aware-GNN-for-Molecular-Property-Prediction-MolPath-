import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class MolPath(nn.Module):
    def __init__(self, input_dim, hidden_dim, K=6, lambda_param=0.1, dropout=0.02):
        super(MolPath, self).__init__()
        self.K = K
        self.lambda_param = lambda_param
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Path Convolution using LSTM [cite: 522]
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Initial Residual Difference Connection (IRDC) [cite: 517, 781]
        # Formula: (1 - λ) * H0 - λ * sum(H_previous)
        
        # Path Attention [cite: 529, 839]
        self.att_weight = nn.Parameter(torch.ones(K + 1))
        
        self.post_conv = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        # Initial node features H(0) [cite: 513]
        h0 = self.embedding(data.x)
        h_layers = [h0]
        
        # Loop through path lengths 1 to K [cite: 528]
        for k in range(1, self.K + 1):
            # 1. Apply IRDC to mitigate redundancy [cite: 484, 521]
            h_prev_sum = torch.stack(h_layers).sum(dim=0)
            h_irdc = (1 - self.lambda_param) * h0 - self.lambda_param * h_prev_sum
            
            # 2. Sequence learning along shortest paths [cite: 522, 525]
            # (Note: This assumes your data object contains pre-computed paths)
            # For simplicity, we simulate the path convolution step here
            h_conv, _ = self.lstm(h_irdc.unsqueeze(1)) 
            h_conv = self.post_conv(h_conv.squeeze(1))
            h_layers.append(h_conv)
            
        # 3. Path Attention [cite: 529, 842]
        # Aggregate initial features and weighted shortest paths
        weights = F.softmax(self.att_weight, dim=0)
        h_final = torch.zeros_like(h0)
        for i in range(len(h_layers)):
            h_final += weights[i] * h_layers[i]
            
        # 4. Global Pooling and Prediction [cite: 529, 846]
        g_repr = global_mean_pool(h_final, data.batch)
        return self.predictor(g_repr)