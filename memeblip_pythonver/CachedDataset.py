from torch.utils.data import Dataset, DataLoader
import torch

class CachedDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.image_features = data['image_features']
        self.text_features = data['text_features']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'image_features': self.image_features[idx],
            'text_features': self.text_features[idx],
            'labels': self.labels[idx]
        }