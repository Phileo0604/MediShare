import json
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

DATASET_PATH = config["DATASET_PATH"]
TARGET_COLUMN = config["TARGET_COLUMN"]
BATCH_SIZE = config["BATCH_SIZE"]

# Custom Dataset Class
class CSVDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target_column: str):
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        self.features = data.drop(columns=['id', target_column], errors='ignore')
        self.labels = data[target_column]

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        self.features = pd.DataFrame(imputer.fit_transform(self.features), columns=self.features.columns)

        # Encode target labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

        # Standardize features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

# Load dataset
def load_datasets():
    data = pd.read_csv(DATASET_PATH)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = CSVDataset(train_data, TARGET_COLUMN)
    test_dataset = CSVDataset(test_data, TARGET_COLUMN)
    return train_dataset, test_dataset

# Prepare DataLoaders
train_dataset, test_dataset = load_datasets()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
