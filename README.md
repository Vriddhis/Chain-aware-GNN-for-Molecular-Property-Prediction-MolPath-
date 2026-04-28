# Advanced Graph Neural Networks for Molecular Property Prediction

## Overview
This repository presents an implementation of **MolPath**, a chain-aware Graph Neural Network (GNN) designed for molecular property prediction, with a primary focus on **Blood-Brain Barrier Penetration (BBBP)**.

Predicting molecular properties is a critical task in drug discovery, where understanding whether a molecule can cross the blood-brain barrier determines its viability for treating central nervous system (CNS) disorders. Traditional Graph Neural Networks (e.g., GCN, MPNN) often struggle to capture **long-range dependencies** within molecular graphs due to limited receptive fields and over-smoothing.

**MolPath** addresses these challenges by explicitly modeling **multi-hop structural interactions** through path-based message passing, enabling richer and more expressive molecular representations.

---

## Key Features
- ✅ Chain-aware message passing via **PathConv layers**
- ✅ Improved long-range dependency modeling (up to 6 hops)
- ✅ Mitigation of over-smoothing using **IRDC**
- ✅ Global context integration with **attention pooling**
- ✅ Hybrid feature space combining **local + global descriptors**

---

## Methodology

### Graph Representation
Molecules are represented as graphs:
- **Nodes:** Atoms with 9-dimensional feature vectors  
- **Edges:** Chemical bonds between atoms  

This representation allows the model to learn structural and chemical relationships directly from molecular topology.

---

### Model Architecture

#### 1. PathConv Layers
The core of MolPath is a stack of **PathConv layers**, which aggregate information across multi-hop neighborhoods. Unlike standard GNN layers that focus on immediate neighbors, PathConv captures **sequential shortest-path information**, enabling the model to learn interactions between distant atoms.

#### 2. Initial Residual Difference Connection (IRDC)
To combat **over-smoothing**, where node representations become indistinguishable in deep GNNs:
- IRDC preserves the **initial node features**
- Introduces a residual correction mechanism  
- Ensures stable and discriminative embeddings across layers  

#### 3. Global Attention Pooling
A **global attention mechanism** is used to aggregate node-level embeddings into a graph-level representation:
- Learns importance weights for each atom  
- Enables adaptive focus on chemically relevant substructures  

#### 4. Feature Augmentation
Node embeddings are enriched with **global molecular descriptors**:
- **MolLogP** (lipophilicity)  
- **Molecular Weight**  

This hybrid approach improves predictive performance by combining:
- Local structural features  
- Global physicochemical properties  

---

## Model Configuration

| Parameter                | Value                         |
|------------------------|------------------------------|
| Node Feature Dimension | 9                            |
| Hidden Dimension       | 256                          |
| Number of Layers       | 6 PathConv layers            |
| Residual Scaling       | Learnable α, fixed λ         |
| Optimizer              | Adam                         |
| Learning Rate          | 5e-4                         |
| Weight Decay           | 1e-4                         |
| Loss Function          | BCEWithLogitsLoss            |
| Evaluation Metric      | ROC-AUC                      |

---

## Dataset

### BBBP (MoleculeNet Benchmark)
- **Task:** Binary classification (BBB permeable vs non-permeable)  
- **Size:** 2,039 valid molecular graphs  
- **Challenge:** Class imbalance  

To ensure robust evaluation, performance is measured using **ROC-AUC**, which is insensitive to class imbalance.

---

## Results

MolPath achieves a **ROC-AUC score of 0.8955**, demonstrating strong performance compared to established baselines:

| Model   | Performance (ROC-AUC) |
|--------|----------------------|
| GCN    | Baseline             |
| MPNN   | Baseline             |
| D-MPNN | Baseline             |
| **MolPath** | **0.8955**     |

### Key Takeaways
- Better capture of **long-range dependencies**
- Reduced over-smoothing in deeper architectures  
- Strong generalization on molecular property prediction tasks  

---

## Implementation Stack

- **PyTorch** – Deep learning framework  
- **PyTorch Geometric** – Graph neural network utilities  
- **RDKit** – Molecular feature extraction and preprocessing  
- **Scikit-learn** – Evaluation metrics and utilities  
---

## Author

**Vriddhi Shetty**  
BTech Artificial Intelligence & Data Science  
K. J. Somaiya School of Engineering  

