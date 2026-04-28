from sklearn.metrics import roc_auc_score
import numpy as np
import torch

def evaluate_rocauc(y_true, y_pred):
    # Convert tensors to numpy and move to CPU
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # ROC-AUC requires at least one positive and one negative sample
    # We check if the number of unique classes is less than 2
    if len(np.unique(y_true)) < 2:
        return 0.5  # Return 0.5 as a neutral baseline
        
    return roc_auc_score(y_true, y_pred)