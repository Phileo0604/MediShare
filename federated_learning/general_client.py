import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict

from federated_learning.dataset_handler import load_dataset
from federated_learning.model_registry import get_model

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")

class FederatedClient:
    def __init__(self, config):
        self.config = config
        # Load dataset using the updated load_dataset function
        self.train_loader, self.test_loader = load_dataset(config)
        
        # Debugging: Print shapes of data loaders
        print(f"Train loader length: {len(self.train_loader.dataset)}")
        print(f"Test loader length: {len(self.test_loader.dataset)}")
        
        # Define the model
        input_dim = self.train_loader.dataset[0][0].shape[0]  # Get input dimension from the first data sample
        output_dim = len(set(self.train_loader.dataset.labels.tolist()))  # Get the number of unique labels
        self.model = get_model(config, input_dim=input_dim, output_dim=output_dim).to(DEVICE)

    def train(self, epochs):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        for _ in range(epochs):
            for X, y in self.train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                output = self.model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                output = self.model(X)
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
