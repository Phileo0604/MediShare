import json
import torch
from model import FeedforwardNN
from dataset import train_loader, train_dataset
from training import train
from utils import export_model_parameters

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

EPOCHS = config["EPOCHS"]

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
input_dim = train_dataset.features.shape[1]
output_dim = len(set(train_dataset.labels))
model = FeedforwardNN(input_dim, output_dim).to(DEVICE)

# Train the model
train(model, train_loader, epochs=EPOCHS)

# Export model parameters
export_model_parameters(model, "model_parameters.json")

print("✅ Model parameters saved successfully. Now you can run main.py for federated learning.")
