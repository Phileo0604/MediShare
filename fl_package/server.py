import flwr as fl
import torch
import os
from model import FeedforwardNN
from dataset import train_dataset
from utils import set_parameters, get_parameters

# Define model checkpoint path
MODEL_PATH = "global_model.pth"

# Function to load or create a global model
def load_model():
    input_dim = train_dataset.features.shape[1]
    output_dim = len(set(train_dataset.labels))

    model = FeedforwardNN(input_dim, output_dim)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("✅ Loaded existing global model.")
    else:
        print("⚡ No existing model found. Initialized a new model.")
    return model

# Function to save the model after each round
def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)
    print("📌 Global model saved.")

# Load the global model
global_model = load_model()
initial_parameters = get_parameters(global_model)  # Get initial model parameters

# Custom FedAvg strategy with persistent model saving
class PersistentFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            # Convert FL parameters to model tensors
            param_tensors = [torch.tensor(param) for param in aggregated_parameters]

            # Set parameters to global model
            set_parameters(global_model, param_tensors)

            # Save model
            save_model(global_model)
            print(f"✅ Round {rnd}: Global model updated and saved.")

        return aggregated_parameters

if __name__ == "__main__":
    print("🚀 Starting Persistent Federated Learning Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),  # Increase rounds for testing
        strategy=PersistentFedAvg(),
        force_final_distributed_eval=False  # Prevents server from waiting on all clients
    )

