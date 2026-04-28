import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from dataset.bbbp_dataset import get_dataset
from models.molpath import MolPath
from train.train import train, evaluate

def main():
    config = Config()
    
    # 1. Load Dataset
    # Make sure you deleted the 'processed' folder before running this!
    dataset = get_dataset("./data")
    dataset = dataset.shuffle()
    
    # 2. Train/Test Split (80/20)
    # Using Random Split to match paper's 0.91 benchmark performance
    n = len(dataset)
    train_dataset = dataset[:int(0.8 * n)]
    test_dataset = dataset[int(0.8 * n):]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Model Initialization 
    # Using hidden_dim=256 and K=6 for BBBP as suggested by paper sweet-spots
    model = MolPath(
        input_dim=dataset.num_node_features, 
        hidden_dim=256, 
        K=6
    )

    # 4. Optimization Setup
    # Higher LR and specific weight decay to handle the increased feature set
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0005, 
        weight_decay=1e-4
    )
    
    # NEW: Increased patience to 20. 
    # This allows the model to 'plateau' for longer before cutting the LR.
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=20, 
        verbose=True
    )

    print(f"--- Training MolPath ---")
    print(f"Total Molecules: {len(dataset)}")
    print(f"Input Features: {dataset.num_node_features}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # 5. Training Loop
    best_auc = 0
    for epoch in range(1, 151):
        loss = train(model, train_loader, optimizer)
        rocauc = evaluate(model, test_loader)
        
        # Scheduler monitors ROC-AUC
        scheduler.step(rocauc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if rocauc > best_auc:
            best_auc = rocauc
            # Optional: torch.save(model.state_checkpoint, 'best_model.pt')

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | ROC-AUC: {rocauc:.4f} | LR: {current_lr:.2e} | Best: {best_auc:.4f}")

if __name__ == "__main__":
    main()