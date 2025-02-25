import json
import torch
from flwr.client import start_client
from dataset import train_loader, test_loader, train_dataset
from model import FeedforwardNN
from utils import import_model_parameters
from federated import FlowerCSVClient

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def client_fn(context):  # 🔹 Fixed: Added `context` parameter
    input_dim = train_dataset.features.shape[1]
    output_dim = len(set(train_dataset.labels))
    model = FeedforwardNN(input_dim, output_dim).to(DEVICE)

    # Load pre-saved model parameters
    import_model_parameters(model, "model_parameters.json")

    return FlowerCSVClient(model, train_loader, test_loader)

if __name__ == "__main__":
    print("🖥️  Starting Federated Learning Client...")
    start_client(
        server_address="127.0.0.1:8080",  # Ensure this matches the server
        client_fn=client_fn  # 🔹 Fixed: Pass `client_fn` instead of `client`
    )
