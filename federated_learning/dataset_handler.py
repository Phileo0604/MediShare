import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from torchvision.datasets import ImageFolder
from PIL import Image

class CSVDataset(Dataset):
    def __init__(self, file_path: str, target_column: str):
        data = pd.read_csv(file_path)
        self.X = data.drop(columns=[target_column]).values
        self.y = data[target_column].values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def load_dataset(config):
    if config["dataset_type"] == "csv":
        dataset = CSVDataset(config["dataset_path"], config["target_column"])
        return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    elif config["dataset_type"] == "image":
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        dataset = ImageFolder(config["dataset_path"], transform=transform)
        return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    else:
        raise ValueError("Unsupported dataset type.")
