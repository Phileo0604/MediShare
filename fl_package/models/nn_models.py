import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from models.xgb_models import XGBoostModel, create_xgboost_model, get_model_path


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
    if hasattr(model, 'get_parameters'):
        # XGBoost model
        return model.get_parameters()
    else:
        # PyTorch model
        return [param.detach().cpu().numpy() for param in model.parameters()]


def set_parameters(model, parameters):
    """Set model parameters from numpy arrays."""
    if hasattr(model, 'set_parameters'):
        # XGBoost model
        model.set_parameters(parameters)
    else:
        # PyTorch model
        with torch.no_grad():
            for param, value in zip(model.parameters(), parameters):
                param.copy_(torch.tensor(value, dtype=param.dtype))


def export_model_parameters(model, file_path="model_parameters.json"):
    """Export model parameters to a file."""
    parameters = get_parameters(model)
    
    # Check if this is an XGBoost model
    if hasattr(model, 'model') and hasattr(model, 'get_parameters'):
        # Use pickle for XGBoost models
        import pickle
        with open(file_path, "wb") as f:
            pickle.dump(parameters, f)
        print(f"XGBoost model parameters saved to {file_path}")
    else:
        # JSON for neural network models
        with open(file_path, "w") as f:
            json.dump([param.tolist() for param in parameters], f)
        print(f"Neural network parameters saved to {file_path}")


def import_model_parameters(model, file_path="model_parameters.json"):
    """Import model parameters from a file."""
    try:
        # Check if this is an XGBoost model
        if hasattr(model, 'model') and hasattr(model, 'set_parameters'):
            # Try to load as pickle (for XGBoost)
            import pickle
            with open(file_path, "rb") as f:
                parameters = pickle.load(f)
            set_parameters(model, parameters)
            print(f"XGBoost model parameters loaded from {file_path}")
        else:
            # Load as JSON (for neural networks)
            with open(file_path, "r") as f:
                parameters = json.load(f)
            set_parameters(model, [np.array(param) for param in parameters])
            print(f"Neural network parameters loaded from {file_path}")
    except FileNotFoundError:
        print(f"Parameter file {file_path} not found. Using default parameters.")
    except Exception as e:
        print(f"Error loading parameters: {e}")
        raise


def create_model(input_dim, output_dim, hidden_layers=None, device=None, dataset_type="breast_cancer", task="classification"):
    """
    Factory function to create and initialize a model based on dataset type.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output classes or values
        hidden_layers: List of hidden layer dimensions
        device: Torch device (CPU or GPU)
        dataset_type: Type of dataset to select appropriate model
        task: 'classification' or 'regression'
    
    Returns:
        Initialized model on the specified device
    """
    if dataset_type.lower() in ["parkinsons", "third_dataset"]:
        # Use XGBoost for these datasets
        if dataset_type.lower() == "parkinsons":
            # Customize XGBoost parameters for Parkinson's dataset
            params = {
                'objective': 'reg:squarederror',  # UPDRS is typically a regression target
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'rmse',
                'tree_method': 'hist',
                'min_child_weight': 3
            }
            task = 'regression'
        else:
            # Default XGBoost parameters for the third dataset
            params = None
        
        return create_xgboost_model(input_dim, output_dim, params=params, task=task)
    else:
        # Use neural network for breast cancer (original implementation)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if hidden_layers is None:
            hidden_layers = [64, 32]
        
        model = FeedforwardNN(input_dim, output_dim, hidden_layers).to(device)
        return model