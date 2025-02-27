import json
import torch
import argparse
from flwr.simulation import run_simulation

# Local imports
from utils.data_utils import load_datasets
from models.nn_models import create_model, export_model_parameters
from training.trainer import train
from federated.client import create_client_app
from federated.server import create_server_app


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


def run_federated_learning(config, train_dataset, test_dataset, train_loader, test_loader, device):
    """Run federated learning simulation."""
    print("\n--- Starting Federated Learning Simulation ---")
    
    # Create server and client applications
    server_app = create_server_app(config)
    client_app = create_client_app(
        config, 
        train_dataset, 
        test_dataset, 
        train_loader, 
        test_loader,
        device
    )
    
    # Run simulation
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=config["federated"]["num_clients"]
    )


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Federated Learning with Neural Networks")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "federated", "both"], default="both",
                        help="Mode to run: train (standalone training only), federated (run FL only), both (default)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize data and model
    train_dataset, test_dataset, train_loader, test_loader, model, device = initialize_data_and_model(config)
    
    # Run in specified mode
    if args.mode in ["train", "both"]:
        run_standalone_training(config, model, train_loader)
        
    if args.mode in ["federated", "both"]:
        run_federated_learning(config, train_dataset, test_dataset, train_loader, test_loader, device)
    
    print("\nDone!")


if __name__ == "__main__":
    main()