from torch.utils.data import Dataset, DataLoader

class SingleCellDataset(Dataset):
    """Single Cell Dataset."""
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

