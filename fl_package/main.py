import json
import torch
import argparse
import os
import uuid
import sys

# Local imports
from utils.data_utils import load_datasets
from models.nn_models import create_model, export_model_parameters
from training.trainer import train


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset class that holds features and labels separately."""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def initialize_data_and_model(config, dataset_type="breast_cancer"):
    """Initialize datasets, data loaders, and model."""
    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets with dataset type
    train_dataset, test_dataset, train_loader, test_loader = load_datasets(
        config["dataset"]["path"],
        config["dataset"]["target_column"],
        config["training"]["batch_size"],
        dataset_type=dataset_type
    )
    
    # Create model
    input_dim = train_dataset.features.shape[1] if hasattr(train_dataset, 'features') else train_dataset[0][0].shape[0]
    
    # Determine output dimension
    if hasattr(train_dataset, 'labels'):
        output_dim = len(set(train_dataset.labels))
    else:
        # Try to infer from loader
        for batch in train_loader:
            output_dim = 1 if len(batch[1].shape) == 1 else batch[1].shape[1]
            break
    
    # Store dimensions in config for the server
    config["model_input_dim"] = input_dim
    config["model_output_dim"] = output_dim
    
    # Get appropriate hidden layers based on dataset type
    hidden_layers = config["model"].get("hidden_layers", [64, 32])
    if dataset_type == "parkinsons" and "parkinsons_hidden_layers" in config["model"]:
        hidden_layers = config["model"]["parkinsons_hidden_layers"]
    elif dataset_type == "third_dataset" and "third_dataset_hidden_layers" in config["model"]:
        hidden_layers = config["model"]["third_dataset_hidden_layers"]
    
    # Determine task type
    task = "regression" if dataset_type == "parkinsons" else "classification"
    
    # Create model with the appropriate architecture
    model = create_model(
        input_dim, 
        output_dim, 
        hidden_layers,
        device,
        dataset_type=dataset_type,
        task=task
    )
    
    return train_dataset, test_dataset, train_loader, test_loader, model, device


def run_standalone_training(config, model, train_loader):
    """Run standalone training to initialize model parameters."""
    print("\n--- Initial Model Training ---")
    train(
        model, 
        train_loader, 
        config["training"]["epochs"],
        config["training"]["learning_rate"]
    )
    
    # Export model parameters
    export_model_parameters(model, config["model"]["parameters_file"])


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Federated Learning with Neural Networks")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["server", "client", "train"], required=True,
                        help="Mode to run: server, client, or train (standalone training only)")
    parser.add_argument("--client-id", type=str, default=None, help="Unique identifier for this client")
    parser.add_argument("--server-host", type=str, default=None, help="Server hostname or IP")
    parser.add_argument("--server-port", type=int, default=None, help="Server port")
    parser.add_argument("--cycles", type=int, default=None, help="Number of client cycles (0 for infinite)")
    parser.add_argument("--wait-time", type=int, default=None, help="Wait time between client cycles in seconds")
    # Add dataset type argument
    parser.add_argument("--dataset-type", type=str, default="breast_cancer",
                        choices=["breast_cancer", "parkinsons", "third_dataset"],
                        help="Type of dataset to use (default: breast_cancer)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate client ID if not provided
    client_id = args.client_id or f"client_{uuid.uuid4().hex[:8]}"
    
    if args.mode == "train":
        # Just train the model and save parameters
        train_dataset, test_dataset, train_loader, test_loader, model, device = initialize_data_and_model(
            config, dataset_type=args.dataset_type
        )
        run_standalone_training(config, model, train_loader)
        
    elif args.mode == "server":
        # For server mode, import from the federated subdirectory
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from federated.server import run_server
        except ModuleNotFoundError as e:
            print(f"Error importing server module: {e}")
            # Try to find where server.py is located
            base_dir = os.path.dirname(os.path.abspath(__file__))
            for root, dirs, files in os.walk(base_dir):
                if "server.py" in files:
                    rel_path = os.path.relpath(root, base_dir)
                    print(f"Found server.py in: {rel_path}")
                    print(f"Try to import using: from {rel_path.replace(os.sep, '.')}.server import run_server")
            raise
        
        # Try to initialize data and model to get dimensions
        try:
            train_dataset, test_dataset, train_loader, test_loader, model, device = initialize_data_and_model(
                config, dataset_type=args.dataset_type
            )
            
            # Save initial model parameters if they don't exist yet
            if not os.path.exists(config["model"]["parameters_file"]):
                export_model_parameters(model, config["model"]["parameters_file"])
                print(f"Initialized model parameters at {config['model']['parameters_file']}")
                
        except Exception as e:
            print(f"Warning: Could not initialize model: {e}")
            print("The server will still start, but may use default dimensions for the model.")
        
        # Run the server
        run_server(config)
        
    elif args.mode == "client":
        # For client mode, import from the federated subdirectory
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from federated.client import run_client
        except ModuleNotFoundError as e:
            print(f"Error importing client module: {e}")
            # Try to find where client.py is located
            base_dir = os.path.dirname(os.path.abspath(__file__))
            for root, dirs, files in os.walk(base_dir):
                if "client.py" in files:
                    rel_path = os.path.relpath(root, base_dir)
                    print(f"Found client.py in: {rel_path}")
                    print(f"Try to import using: from {rel_path.replace(os.sep, '.')}.client import run_client")
            raise
        
        # Get dataset-specific configurations
        dataset_path = config["dataset"].get("path")
        target_column = config["dataset"].get("target_column")
        
        # Use dataset-specific path and target column if available
        if args.dataset_type == "parkinsons":
            if "parkinsons_path" in config["dataset"]:
                dataset_path = config["dataset"]["parkinsons_path"]
            if "parkinsons_target_column" in config["dataset"]:
                target_column = config["dataset"]["parkinsons_target_column"]
        elif args.dataset_type == "third_dataset":
            if "third_dataset_path" in config["dataset"]:
                dataset_path = config["dataset"]["third_dataset_path"]
            if "third_dataset_target_column" in config["dataset"]:
                target_column = config["dataset"]["third_dataset_target_column"]
        
        # Update config with dataset-specific settings
        config["dataset"]["path"] = dataset_path
        config["dataset"]["target_column"] = target_column
        
        # Initialize data and model
        train_dataset, test_dataset, train_loader, test_loader, model, device = initialize_data_and_model(
            config, dataset_type=args.dataset_type
        )
        
        # Get server details from args or config
        # For client connections, use client_host (usually 127.0.0.1 or the server's actual IP)
        server_host = args.server_host or config["server"].get("client_host", "127.0.0.1")
        server_port = args.server_port or config["server"].get("port", 8080)
        cycles = args.cycles if args.cycles is not None else config["client"].get("cycles", 0)
        wait_time = args.wait_time if args.wait_time is not None else config["client"].get("wait_time", 60)
        
        print(f"Client will connect to server at {server_host}:{server_port}")
        print(f"Using dataset type: {args.dataset_type}")
        
        # Run client
        run_client(
            config=config,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            client_id=client_id,
            server_host=server_host,
            server_port=server_port,
            num_cycles=cycles,
            wait_time=wait_time,
            dataset_type=args.dataset_type  # Pass dataset type to run_client
        )


if __name__ == "__main__":
    main()