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


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def initialize_data_and_model(config):
    """Initialize datasets, data loaders, and model."""
    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset, test_dataset, train_loader, test_loader = load_datasets(
        config["dataset"]["path"],
        config["dataset"]["target_column"],
        config["training"]["batch_size"]
    )
    
    # Create model
    input_dim = train_dataset.features.shape[1]
    output_dim = len(set(train_dataset.labels))
    
    # Store dimensions in config for the server
    config["model_input_dim"] = input_dim
    config["model_output_dim"] = output_dim
    
    model = create_model(
        input_dim, 
        output_dim, 
        config["model"]["hidden_layers"],
        device
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
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate client ID if not provided
    client_id = args.client_id or f"client_{uuid.uuid4().hex[:8]}"
    
    if args.mode == "train":
        # Just train the model and save parameters
        train_dataset, test_dataset, train_loader, test_loader, model, device = initialize_data_and_model(config)
        run_standalone_training(config, model, train_loader)
        
    elif args.mode == "server":
        # For continuous server mode, import here to avoid unnecessary imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from federated.continuous_server import run_server
        
        # Try to initialize data and model to get dimensions
        try:
            train_dataset, test_dataset, train_loader, test_loader, model, device = initialize_data_and_model(config)
            
            # Save initial model parameters if they don't exist yet
            if not os.path.exists(config["model"]["parameters_file"]):
                export_model_parameters(model, config["model"]["parameters_file"])
                print(f"Initialized model parameters at {config['model']['parameters_file']}")
                
        except Exception as e:
            print(f"Warning: Could not initialize model: {e}")
            print("The server will still start, but may use default dimensions for the model.")
        
        # Run the continuous server
        run_server(config)
        
    elif args.mode == "client":
        # For continuous client mode, import here to avoid unnecessary imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from federated.continuous_client import run_client
        
        # Initialize data and model
        train_dataset, test_dataset, train_loader, test_loader, model, device = initialize_data_and_model(config)
        
        # Get server details from args or config
        # For client connections, use client_host (usually 127.0.0.1 or the server's actual IP)
        server_host = args.server_host or config["server"].get("client_host", "127.0.0.1")
        server_port = args.server_port or config["server"].get("port", 8080)
        cycles = args.cycles if args.cycles is not None else config["client"].get("cycles", 0)
        wait_time = args.wait_time if args.wait_time is not None else config["client"].get("wait_time", 60)
        
        print(f"Client will connect to server at {server_host}:{server_port}")
        
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
            wait_time=wait_time
        )


if __name__ == "__main__":
    main()