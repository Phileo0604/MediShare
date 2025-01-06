import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict

from federated_learning.dataset_handler import load_dataset
from federated_learning.model_registry import get_model
from federated_learning_practice.dataloader import DEVICE

class FederatedClient:
    def __init__(self, config):
        self.config = config
        self.train_loader = load_dataset(config)
        self.model = get_model(config, input_dim=self.train_loader.dataset[0][0].shape[0],
                               output_dim=len(set(self.train_loader.dataset.dataset.y.tolist()))).to(DEVICE)

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
