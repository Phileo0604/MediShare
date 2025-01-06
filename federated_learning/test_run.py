import json
import pandas as pd
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from typing import List, Tuple

# Flower imports
import flwr
from flwr.client import NumPyClient, Client, ClientApp
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

# Additional utilities
from datasets.utils.logging import disable_progress_bar


# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# Load configuration from config.json
dataset_path = config["dataset_path"]
target_column = config["target_column"]
model_type = config["model_type"]
epochs = config["epochs"]
batch_size = config["batch_size"]

# Additional configurations
features_columns = config["features_columns"]
train_partition = config["partitioners"]["train"]
test_partition = config["partitioners"]["test"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.preprocessing import LabelEncoder

class CSVDataset(Dataset):
    def __init__(self, data, target_column, transform=None):
        # Drop 'Unnamed: 32' or any other irrelevant columns
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        
        self.data = data
        self.target_column = target_column
        self.transform = transform
        
        # Drop 'id' and target column 'diagnosis' from features
        self.features = self.data.drop(columns=['id', self.target_column])
        self.labels = self.data[self.target_column]
        
        # Handle missing values (fill with column mean)
        imputer = SimpleImputer(strategy='mean')
        self.features = pd.DataFrame(imputer.fit_transform(self.features), columns=self.features.columns)
        
        # Encode target labels ('M' -> 1, 'B' -> 0)
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)  # "M" -> 1, "B" -> 0
        
        # Standardize the feature columns (numerical values)
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label
     

# Load dataset
train_data, test_data = train_test_split(pd.read_csv(dataset_path), test_size=0.2)
train_dataset = CSVDataset(train_data, target_column)
test_dataset = CSVDataset(test_data, target_column)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

def get_parameters(model):
    return [tensor.cpu().numpy() for tensor in model.parameters()]

def set_parameters(model, parameters):
    with torch.no_grad():
        for param, value in zip(model.parameters(), parameters):
            param.copy_(torch.tensor(value, dtype=param.dtype))

def train(model, train_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            output = model(features)
            loss = criterion(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy


class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class FlowerCSVClient(NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train(self.model, self.train_loader, epochs=epochs)
        return get_parameters(self.model), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader)
        return float(loss), len(self.test_loader), {"accuracy": float(accuracy)}

    
def client_fn(context: Context) -> Client:
    # Load model based on config
    input_dim = len(pd.read_csv(dataset_path).drop(columns=[target_column]).columns)
    output_dim = len(pd.read_csv(dataset_path)[target_column].unique())
    model = FeedforwardNN(input_dim, output_dim).to(DEVICE)

    # Load CSV data and create DataLoader
    train_loader = DataLoader(CSVDataset(pd.read_csv(dataset_path), target_column), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CSVDataset(pd.read_csv(dataset_path), target_column), batch_size=batch_size)
    
    return FlowerCSVClient(model, train_loader, test_loader)
