class Config:
    def __init__(self):
        self.batch_size = 32      # Paper uses 16, 32, 64, or 128 
        self.lr = 1e-4            # Fixed at 10^-4 in the paper 
        self.hidden_dim = 256     # Paper uses between 128 and 700 
        self.epochs = 150
        self.K = 6                # Shortest path length (Paper: 4 to 12) 
        self.lambda_param = 0.1   # IRDC ratio (Paper: 0 to 0.6) 
        self.dropout = 0.02       # Paper: 0 to 0.03