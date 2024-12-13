import torch
from torch.utils.data import DataLoader, Dataset

class KaloDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_data_loader(data, labels, batch_size=32, shuffle=True):
    dataset = KaloDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
