import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np


class FeedforwardNN(nn.Module):
    """Simple feedforward neural network."""
    
    def __init__(self, input_dim, output_dim, hidden_layers=None):
        super(FeedforwardNN, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [64, 32]
            
        # First layer
        layers = [nn.Linear(input_dim, hidden_layers[0])]
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        # Create sequential model
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        # Apply layers with ReLU activation except for the last layer
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        
        # Apply the output layer without activation (for CrossEntropyLoss)
        x = self.layers[-1](x)
        return x


def get_parameters(model):
    """Extract model parameters as numpy arrays."""
    return [param.detach().cpu().numpy() for param in model.parameters()]


def set_parameters(model, parameters):
    """Set model parameters from numpy arrays."""
    with torch.no_grad():
        for param, value in zip(model.parameters(), parameters):
            param.copy_(torch.tensor(value, dtype=param.dtype))


def export_model_parameters(model, file_path="model_parameters.json"):
    """Export model parameters to a JSON file."""
    parameters = get_parameters(model)
    with open(file_path, "w") as f:
        json.dump([param.tolist() for param in parameters], f)
    print(f"Model parameters saved to {file_path}")


def import_model_parameters(model, file_path="model_parameters.json"):
    """Import model parameters from a JSON file."""
    try:
        with open(file_path, "r") as f:
            parameters = json.load(f)
        set_parameters(model, [np.array(param) for param in parameters])
        print(f"Model parameters loaded from {file_path}")
    except FileNotFoundError:
        print(f"Parameter file {file_path} not found. Using default parameters.")
    except Exception as e:
        print(f"Error loading parameters: {e}")
        raise


def create_model(input_dim, output_dim, hidden_layers=None, device=None):
    """Factory function to create and initialize a model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FeedforwardNN(input_dim, output_dim, hidden_layers).to(device)
    return model