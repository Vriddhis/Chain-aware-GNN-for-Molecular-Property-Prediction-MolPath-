Advanced Graph Neural Networks for Molecular Property Prediction
Overview

This repository presents an implementation of MolPath, a chain-aware Graph Neural Network designed for molecular property prediction, with a focus on Blood-Brain Barrier Penetration (BBBP). The model addresses limitations of conventional GNNs in capturing long-range structural dependencies within molecular graphs.

Methodology

Molecules are represented as graphs with atoms as nodes and bonds as edges. The architecture integrates:

PathConv Layers for modeling multi-hop atomic interactions
Initial Residual Difference Connection (IRDC) to mitigate over-smoothing and preserve initial node representations
Global Attention Pooling for adaptive aggregation of atom-level embeddings
Augmented node features with global molecular descriptors (MolLogP, Molecular Weight)
Model Configuration
Node feature dimension: 9
Hidden dimension: 256
Depth: 6 PathConv layers
Residual scaling: learnable α, fixed λ
Optimizer: Adam (lr = 5e-4, weight decay = 1e-4)
Loss: BCEWithLogitsLoss
Evaluation metric: ROC-AUC
Dataset
BBBP (MoleculeNet benchmark)
2,039 valid molecular graphs
Binary classification with class imbalance handled via ROC-AUC
Results

The model achieves a ROC-AUC of 0.8955, demonstrating competitive performance relative to established GNN baselines such as GCN, MPNN, and D-MPNN.

Implementation Stack
PyTorch
PyTorch Geometric
RDKit
Scikit-learn

Future Directions
Explicit shortest-path sequence extraction
Integration with transformer-based architectures
Incorporation of 3D geometric features
Multi-task molecular property prediction
Author

Vriddhi Shetty
BTech Artificial Intelligence & Data Science
K. J. Somaiya School of Engineering
