from dataset.bbbp_dataset import BBBPDataset

def get_dataset(root, cutoff, path_type):
    # simplified version (no path logic yet)
    return BBBPDataset(root=root)