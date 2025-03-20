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

def load_config(config_path="config_files/server_config.json"):
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
    elif dataset_type == "reinopath" and "reinopath_hidden_layers" in config["model"]:
        hidden_layers = config["model"]["reinopath_hidden_layers"]
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

def normalize_config(config):
    """
    Normalize the configuration to ensure backward compatibility.
    
    This function transforms new format configs into a structure that
    the existing code can work with.
    """
    normalized = {}
    
    # Check if this is a nested configuration from the new format
    if "configData" in config:
        # Extract data from the nested structure
        config_data = config["configData"]
        
        # Copy the dataset type
        normalized["dataset_type"] = config.get("datasetType", "breast_cancer")
        
        # Ensure minimum required structure exists
        normalized["dataset"] = {}
        normalized["training"] = {"epochs": 10, "batch_size": 32, "learning_rate": 0.001}
        normalized["model"] = {}
        normalized["server"] = {}
        normalized["client"] = {}
        
        # Copy any existing sections
        if "dataset" in config_data:
            normalized["dataset"] = config_data["dataset"]
        
        if "model" in config_data:
            normalized["model"] = config_data["model"]
            
            # If no parameters_file is specified, set a default based on the dataset type
            if "parameters_file" not in normalized["model"]:
                if normalized["dataset_type"] == "breast_cancer":
                    normalized["model"]["parameters_file"] = "global_models/breast_cancer_model.json"
                elif normalized["dataset_type"] == "parkinsons":
                    normalized["model"]["parameters_file"] = "global_models/parkinsons_model.pkl"
                elif normalized["dataset_type"] == "reinopath":
                    normalized["model"]["parameters_file"] = "global_models/reinopath_model.pkl"
                else:
                    normalized["model"]["parameters_file"] = f"global_models/{normalized['dataset_type']}_model.json"
        
        if "server" in config_data:
            normalized["server"] = config_data["server"]
        
        if "client" in config_data:
            normalized["client"] = config_data["client"]
        
        # If dataset info is missing, create default paths
        if not normalized["dataset"].get("path"):
            normalized["dataset"]["path"] = f"datasets/{normalized['dataset_type']}_data.csv"
        
        if not normalized["dataset"].get("target_column"):
            if normalized["dataset_type"] == "breast_cancer":
                normalized["dataset"]["target_column"] = "diagnosis"
            elif normalized["dataset_type"] == "parkinsons":
                normalized["dataset"]["target_column"] = "UPDRS"
            elif normalized["dataset_type"] == "reinopath":
                normalized["dataset"]["target_column"] = "class"
            else:
                normalized["dataset"]["target_column"] = "target"
    else:
        # Old format - just return as is
        normalized = config
    
    return normalized

def run_parameter_only_client(
    config,
    model,
    client_id=None,
    server_host="127.0.0.1",
    server_port=8080,
    num_cycles=1,
    wait_time=10,
    dataset_type="breast_cancer"
):
    """Run a federated learning client in parameter-only mode (no dataset/training)."""
    from federated.client import Client
    
    # Create client with None for train_loader and test_loader
    client = Client(
        model=model,
        train_loader=None,
        test_loader=None,
        epochs=1,  # Not used but required by the API
        client_id=client_id,
        server_host=server_host,
        server_port=server_port,
        dataset_type=dataset_type,
        skip_training=True  # Always skip training in this mode
    )
    
    # Start continuous learning
    print("Starting parameter-only client (no dataset, just parameter sharing)")
    client.start_continuous_learning(num_cycles=num_cycles, wait_time=wait_time)

def create_minimal_model(config, dataset_type):
    """Create a minimal model for parameter-only mode."""
    from models.nn_models import create_model
    
    # Get model info from config if available
    model_config = config.get("model", {})
    if "configData" in config and "model" in config["configData"]:
        model_config = config["configData"]["model"]
    
    # Default dimensions based on dataset type
    input_dim = 30  # Default
    output_dim = 2  # Default binary classification
    
    if dataset_type == "breast_cancer":
        input_dim = 30
        output_dim = 2
    elif dataset_type == "parkinsons":
        input_dim = 16
        output_dim = 1  # Regression
    elif dataset_type == "reinopath":
        input_dim = 19
        output_dim = 2
    
    # Get hidden layers from config
    hidden_layers = model_config.get("hidden_layers", [64, 32])
    
    # Determine task
    task = "regression" if dataset_type.lower() == "parkinsons" else "classification"
    
    # Create model
    model = create_model(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        dataset_type=dataset_type,
        task=task
    )
    
    return model

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
    parser.add_argument("--dataset-type", type=str, default="breast_cancer",
                        choices=["breast_cancer", "parkinsons", "reinopath", "third_dataset"],
                        help="Type of dataset to use")
    # Add new arguments
    parser.add_argument("--skip-training", action="store_true", help="Skip training and only evaluate model")
    parser.add_argument("--parameter-path", type=str, help="Path to pre-trained model parameters")
    # Add new argument for skip-dataset
    parser.add_argument("--skip-dataset", action="store_true", 
                       help="Skip loading dataset (use with --parameter-path for model-only mode)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Normalize the configuration for backwards compatibility
    config = normalize_config(config)
    
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
        
        # Check if we should run in dataset-free mode
        dataset_free_mode = args.skip_training and args.parameter_path
        
        if dataset_free_mode:
            print(f"Running in dataset-free mode with pre-trained parameters from {args.parameter_path}")
            
            # Create a minimal model with default dimensions based on dataset type
            input_dim = 30  # Default
            output_dim = 2  # Default binary classification
            
            if args.dataset_type == "breast_cancer":
                input_dim = 30
                output_dim = 2
            elif args.dataset_type == "parkinsons":
                input_dim = 16
                output_dim = 1  # Regression
            elif args.dataset_type == "reinopath":
                input_dim = 19
                output_dim = 2
            
            # Get model config section
            model_config = config.get("model", {})
            if "configData" in config and "model" in config["configData"]:
                model_config = config["configData"]["model"]
            
            # Get hidden layers from config
            hidden_layers = model_config.get("hidden_layers", [64, 32])
            
            # Determine task type
            task = "regression" if args.dataset_type == "parkinsons" else "classification"
            
            # Import the create_model function
            from models.nn_models import create_model, import_model_parameters
            
            # Create model with appropriate dimensions
            model = create_model(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_layers=hidden_layers,
                dataset_type=args.dataset_type,
                task=task
            )
            
            # Load pre-trained parameters
            try:
                import_model_parameters(model, args.parameter_path)
                print(f"Loaded pre-trained parameters from {args.parameter_path}")
            except Exception as e:
                print(f"Failed to load parameters from {args.parameter_path}: {e}")
                sys.exit(1)  # Exit if parameters can't be loaded in parameter-only mode
            
            # Get server details from args or config
            server_host = args.server_host or config["server"].get("client_host", "127.0.0.1")
            server_port = args.server_port or config["server"].get("port", 8080)
            cycles = args.cycles if args.cycles is not None else config["client"].get("cycles", 0)
            wait_time = args.wait_time if args.wait_time is not None else config["client"].get("wait_time", 60)
            
            print(f"Client will connect to server at {server_host}:{server_port}")
            print(f"Using dataset type: {args.dataset_type}")
            print(f"Operating in parameter-sharing mode (no dataset, no training)")
            
            # Run client with None for datasets and loaders
            run_client(
                config=config,
                train_dataset=None,
                test_dataset=None,
                train_loader=None,
                test_loader=None,
                model=model,
                client_id=client_id,
                server_host=server_host,
                server_port=server_port,
                num_cycles=cycles,
                wait_time=wait_time,
                dataset_type=args.dataset_type,
                skip_training=True
            )
        else:
            # Regular mode with dataset
            # Get dataset-specific configurations
            dataset_path = config["dataset"].get("path")
            target_column = config["dataset"].get("target_column")
            
            # Use dataset-specific path and target column if available
            if args.dataset_type == "parkinsons":
                if "parkinsons_path" in config.get("dataset", {}):
                    dataset_path = config["dataset"]["parkinsons_path"]
                if "parkinsons_target_column" in config.get("dataset", {}):
                    target_column = config["dataset"]["parkinsons_target_column"]
            elif args.dataset_type == "reinopath":
                if "reinopath_path" in config.get("dataset", {}):
                    dataset_path = config["dataset"]["reinopath_path"]
                if "reinopath_target_column" in config.get("dataset", {}):
                    target_column = config["dataset"]["reinopath_target_column"]
            elif args.dataset_type == "third_dataset":
                if "third_dataset_path" in config.get("dataset", {}):
                    dataset_path = config["dataset"]["third_dataset_path"]
                if "third_dataset_target_column" in config.get("dataset", {}):
                    target_column = config["dataset"]["third_dataset_target_column"]
            
            # Create default dataset settings if they don't exist
            if "dataset" not in config:
                config["dataset"] = {
                    "path": dataset_path,
                    "target_column": target_column
                }
            
            # Update config with dataset-specific settings
            config["dataset"]["path"] = dataset_path
            config["dataset"]["target_column"] = target_column
            
            # Initialize data and model
            train_dataset, test_dataset, train_loader, test_loader, model, device = initialize_data_and_model(
                config, dataset_type=args.dataset_type
            )
            
            # Load parameters if path is provided
            if args.parameter_path:
                try:
                    from models.nn_models import import_model_parameters
                    import_model_parameters(model, args.parameter_path)
                    print(f"Loaded pre-trained parameters from {args.parameter_path}")
                except Exception as e:
                    print(f"Failed to load parameters from {args.parameter_path}: {e}")
            
            # Get server details from args or config
            server_host = args.server_host or config["server"].get("client_host", "127.0.0.1")
            server_port = args.server_port or config["server"].get("port", 8080)
            cycles = args.cycles if args.cycles is not None else config["client"].get("cycles", 0)
            wait_time = args.wait_time if args.wait_time is not None else config["client"].get("wait_time", 60)
            
            print(f"Client will connect to server at {server_host}:{server_port}")
            print(f"Using dataset type: {args.dataset_type}")
            print(f"Training will be {'skipped' if args.skip_training else 'performed'}")
            
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
                dataset_type=args.dataset_type,
                skip_training=args.skip_training
            )


if __name__ == "__main__":
    main()