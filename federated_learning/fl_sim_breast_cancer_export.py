import json
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Flower imports
import flwr
from flwr.client import NumPyClient, Client, ClientApp
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)

dataset_path = config["dataset_path"]
target_column = config["target_column"]
epochs = config["epochs"]
batch_size = config["batch_size"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Dataset class
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
    data = pd.read_csv(dataset_path)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = CSVDataset(train_data, target_column)
    test_dataset = CSVDataset(test_data, target_column)
    return train_dataset, test_dataset

train_dataset, test_dataset = load_datasets()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Neural network definition
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

# Model helpers
def get_parameters(model):
    return [param.detach().cpu().numpy() for param in model.parameters()]

def set_parameters(model, parameters):
    with torch.no_grad():
        for param, value in zip(model.parameters(), parameters):
            param.copy_(torch.tensor(value, dtype=param.dtype))

# Export model parameters to a file
def export_model_parameters(model, file_path="model_parameters.json"):
    parameters = get_parameters(model)
    with open(file_path, "w") as f:
        json.dump([param.tolist() for param in parameters], f)
    print(f"Model parameters saved to {file_path}")

# Import model parameters from a file
def import_model_parameters(model, file_path="model_parameters.json"):
    with open(file_path, "r") as f:
        parameters = json.load(f)
    set_parameters(model, [torch.tensor(param) for param in parameters])
    print(f"Model parameters loaded from {file_path}")

# Training and evaluation
def train(model, train_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

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
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

# Flower client
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
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

# Client function for federated learning
def client_fn(context: Context):
    input_dim = train_dataset.features.shape[1]
    output_dim = len(set(train_dataset.labels))
    model = FeedforwardNN(input_dim, output_dim).to(DEVICE)

    # Load the pre-saved parameters
    import_model_parameters(model, "model_parameters.json")

    return FlowerCSVClient(model, train_loader, test_loader)

# Server configuration
def server_fn(context):
    strategy = FedAvg()
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)

server = ServerApp(server_fn=server_fn)
client = ClientApp(client_fn=client_fn)

# Train model and export parameters
input_dim = train_dataset.features.shape[1]
output_dim = len(set(train_dataset.labels))
model = FeedforwardNN(input_dim, output_dim).to(DEVICE)
train(model, train_loader, epochs=epochs)
export_model_parameters(model, "model_parameters.json")

# Run the simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=5  # Number of participating clients
)
