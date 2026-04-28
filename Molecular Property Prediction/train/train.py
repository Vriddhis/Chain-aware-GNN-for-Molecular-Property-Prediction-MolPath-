import torch
import torch.nn.functional as F
from utils.metrics import evaluate_rocauc

def train(model, loader, optimizer):
    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        
        # MolPath forward pass
        out = model(data) 
        
        # Ensure target is (batch_size, 1) and float for BCE loss
        target = data.y.view(-1, 1).float()
        
        loss = F.binary_cross_entropy_with_logits(out, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for data in loader:
            out = model(data)
            
            # Store true labels
            ys.append(data.y.view(-1, 1))
            
            # IMPORTANT: Convert logits to probabilities (0 to 1) for ROC-AUC
            preds.append(torch.sigmoid(out))

    # Concatenate all batches
    ys = torch.cat(ys, dim=0)
    preds = torch.cat(preds, dim=0)

    return evaluate_rocauc(ys, preds)