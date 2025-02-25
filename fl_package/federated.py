import json
import torch
from flwr.client import NumPyClient
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from training import train, test
from model import FeedforwardNN
from dataset import train_loader, test_loader, train_dataset
from utils import get_parameters, set_parameters, import_model_parameters

with open("config.json", "r") as f:
    config = json.load(f)

EPOCHS = config["EPOCHS"]
NUM_ROUNDS = config["NUM_ROUNDS"]

class FlowerCSVClient(NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train(self.model, self.train_loader, epochs=EPOCHS)
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

def client_fn(context):
    input_dim = train_dataset.features.shape[1]
    output_dim = len(set(train_dataset.labels))
    model = FeedforwardNN(input_dim, output_dim)
    import_model_parameters(model, "model_parameters.json")
    return FlowerCSVClient(model, train_loader, test_loader)

def server_fn(context):
    strategy = FedAvg()
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)
