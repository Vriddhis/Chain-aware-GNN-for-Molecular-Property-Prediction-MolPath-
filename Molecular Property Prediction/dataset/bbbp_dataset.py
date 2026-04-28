import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem import Descriptors

def get_atom_features(atom):
    # Added LogP-contribution and Molar Refractivity indicators
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetFormalCharge(),
        1.0 if atom.GetIsAromatic() else 0.0,
        [Chem.rdchem.HybridizationType.SP, 
         Chem.rdchem.HybridizationType.SP2, 
         Chem.rdchem.HybridizationType.SP3].index(atom.GetHybridization()) 
         if atom.GetHybridization() in [Chem.rdchem.HybridizationType.SP, 
         Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3] else 3,
        1.0 if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0
    ]
    return features

class BBBPDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self): return ["BBBP.csv"]
    @property
    def processed_file_names(self): return ["data.pt"]

    def process(self):
        path = self.raw_paths[0]
        df = pd.read_csv(path)
        if "p_np" in df.columns: df = df.rename(columns={"p_np": "label"})
        df = df.dropna(subset=["smiles", "label"])
        
        data_list = []
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None: continue
            
            # Global Molecular Features (LogP and MW)
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            x = torch.tensor([get_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
            
            # Append global context to every atom
            global_feats = torch.tensor([[logp, mw]] * x.size(0))
            x = torch.cat([x, global_feats], dim=-1)

            edges = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edges.append([i, j]); edges.append([j, i])
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            data = Data(x=x, edge_index=edge_index, y=torch.tensor([row["label"]], dtype=torch.float))
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def get_dataset(root="./data"):
    return BBBPDataset(root=root)