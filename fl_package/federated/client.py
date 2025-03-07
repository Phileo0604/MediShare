import socket
import json
import numpy as np
import torch
import logging
import pickle
from typing import List, Dict, Tuple, Optional
import time
import os

from models.nn_models import get_parameters, set_parameters, get_model_path, create_model
from training.trainer import train, test

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Client")


class Client:
    """
    A federated learning client that connects to a server,
    receives the current global model, trains locally, and sends updates.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        epochs: int = 1,
        client_id: str = None,
        server_host: str = "127.0.0.1",
        server_port: int = 8080,
        retry_interval: int = 10,  # Seconds to wait between retries
        dataset_type: str = "breast_cancer"  # Type of dataset
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.client_id = client_id or "unknown"
        self.server_host = server_host
        self.server_port = server_port
        self.retry_interval = retry_interval
        self.dataset_type = dataset_type
        
        # Determine if this is an XGBoost model
        self.is_xgboost = hasattr(self.model, 'model') and hasattr(self.model, 'get_parameters')
        
        # Get appropriate model path based on dataset type
        self.model_path = get_model_path(dataset_type)
        logger.info(f"Using model path: {self.model_path}")
    
    def connect_to_server(self) -> Optional[socket.socket]:
        """Connect to the server."""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.server_host, self.server_port))
            return client_socket
        except Exception as e:
            logger.error(f"Failed to connect to server {self.server_host}:{self.server_port}: {e}")
            return None
    
    def receive_global_model(self, client_socket: socket.socket) -> bool:
        """Receive global model parameters from the server."""
        try:
            # Send dataset type to server so it can send the right model
            dataset_type_bytes = self.dataset_type.encode('utf-8')
            size_bytes = len(dataset_type_bytes).to_bytes(8, byteorder='big')
            client_socket.sendall(size_bytes)
            client_socket.sendall(dataset_type_bytes)
            logger.info(f"Sent dataset type: {self.dataset_type}")
            
            # First get the size of incoming data
            size_bytes = client_socket.recv(8)
            if not size_bytes:
                logger.warning("Server disconnected before sending model size")
                return False
                
            data_size = int.from_bytes(size_bytes, byteorder='big')
            
            # If data_size is 0, server has no model yet
            if data_size == 0:
                logger.info(f"Server has no global model for {self.dataset_type} yet")
                return True
            
            logger.info(f"Expecting {data_size} bytes from server")
            
            # Receive all data
            received_data = b''
            while len(received_data) < data_size:
                chunk = client_socket.recv(min(4096, data_size - len(received_data)))
                if not chunk:
                    break
                received_data += chunk
            
            # Parse received parameters
            if len(received_data) == data_size:
                if self.is_xgboost:
                    # Parse as pickle for XGBoost
                    global_parameters = pickle.loads(received_data)
                else:
                    # Parse as JSON for neural networks
                    parameters_json = received_data.decode('utf-8')
                    global_parameters = [np.array(param) for param in json.loads(parameters_json)]
                
                # Update model with global parameters
                set_parameters(self.model, global_parameters)
                logger.info(f"Received and set global model parameters for {self.dataset_type}")
                return True
            else:
                logger.warning(f"Incomplete data from server")
                return False
        
        except Exception as e:
            logger.error(f"Error receiving global model: {e}")
            return False
    
    def send_model_update(self, client_socket: socket.socket) -> bool:
        """Send updated model parameters to the server."""
        try:
            # Get current model parameters
            parameters = get_parameters(self.model)
            
            if self.is_xgboost:
                # Use pickle for XGBoost models
                parameters_bytes = pickle.dumps(parameters)
            else:
                # Use JSON for neural network models
                parameters_json = json.dumps([param.tolist() for param in parameters])
                parameters_bytes = parameters_json.encode('utf-8')
            
            # Send the size of the data first
            size_bytes = len(parameters_bytes).to_bytes(8, byteorder='big')
            client_socket.sendall(size_bytes)
            
            # Send the parameters
            client_socket.sendall(parameters_bytes)
            logger.info(f"Sent updated model parameters ({len(parameters_bytes)} bytes) for {self.dataset_type}")
            
            # Wait for acknowledgment
            ack = client_socket.recv(3)
            if ack == b'ACK':
                logger.info("Server acknowledged receipt of parameters")
                return True
            else:
                logger.warning(f"Unexpected acknowledgment from server: {ack}")
                return False
        
        except Exception as e:
            logger.error(f"Error sending model update: {e}")
            return False
    
    def train_local_model(self) -> None:
        """Train the model on local data."""
        logger.info(f"Starting local training for {self.epochs} epochs on {self.dataset_type} dataset")
        train(self.model, self.train_loader, epochs=self.epochs)
        
        # Evaluate after training
        loss, accuracy = test(self.model, self.test_loader)
        logger.info(f"Local evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def run_federated_cycle(self) -> bool:
        """Run a complete federated learning cycle."""
        success = False
        
        # Connect to server
        client_socket = self.connect_to_server()
        if not client_socket:
            return False
        
        try:
            # Get global model from server
            if not self.receive_global_model(client_socket):
                return False
            
            # Train model locally
            self.train_local_model()
            
            # Send updated model to server
            success = self.send_model_update(client_socket)
            
        except Exception as e:
            logger.error(f"Error during federated cycle: {e}")
            success = False
        finally:
            client_socket.close()
        
        return success
    
    def start_continuous_learning(self, num_cycles: int = 1, wait_time: int = 10) -> None:
        """
        Start continuous federated learning process.
        
        Args:
            num_cycles: Number of cycles to run (0 for infinite)
            wait_time: Time to wait between cycles in seconds
        """
        cycle_count = 0
        
        while num_cycles == 0 or cycle_count < num_cycles:
            cycle_count += 1
            logger.info(f"Starting federated cycle {cycle_count} for {self.dataset_type}")
            
            success = self.run_federated_cycle()
            
            if success:
                logger.info(f"Federated cycle {cycle_count} completed successfully")
            else:
                logger.warning(f"Federated cycle {cycle_count} failed")
                # Wait before retry
                time.sleep(self.retry_interval)
                continue
            
            if num_cycles == 0 or cycle_count < num_cycles:
                logger.info(f"Waiting {wait_time} seconds before next cycle")
                time.sleep(wait_time)


def run_client(
    config,
    train_dataset,
    test_dataset,
    train_loader,
    test_loader,
    model=None,
    client_id=None,
    server_host="127.0.0.1",
    server_port=8080,
    num_cycles=1,
    wait_time=10,
    dataset_type="breast_cancer"
):
    """Run a federated learning client with specified dataset type."""
    # Create model if not provided
    if model is None:
        # Determine task type based on dataset
        task = "regression" if dataset_type.lower() == "parkinsons" else "classification"
        
        # Determine input dimension based on dataset
        if hasattr(train_dataset, 'features'):
            input_dim = train_dataset.features.shape[1]
        elif hasattr(train_dataset, 'tensors') and len(train_dataset.tensors) > 0:
            input_dim = train_dataset.tensors[0].shape[1]
        else:
            # Try to get dimension from first batch
            for features, _ in train_loader:
                if hasattr(features, 'shape'):
                    input_dim = features.shape[1]
                    break
        
        # Determine output dimension based on dataset type and structure
        if hasattr(train_dataset, 'labels'):
            if task == "regression":
                output_dim = 1
            else:
                output_dim = len(set(train_dataset.labels))
        elif hasattr(train_dataset, 'tensors') and len(train_dataset.tensors) > 1:
            if train_dataset.tensors[1].dim() > 1:
                output_dim = train_dataset.tensors[1].shape[1]  # One-hot encoded
            else:
                if task == "regression":
                    output_dim = 1
                else:
                    output_dim = len(torch.unique(train_dataset.tensors[1]))  # Class indices
        else:
            # Default values
            if task == "regression":
                output_dim = 1
            else:
                output_dim = 2  # Binary classification
        
        # Get appropriate hidden layers from config based on dataset type
        hidden_layers = config["model"].get("hidden_layers", [64, 32])
        if dataset_type.lower() == "parkinsons":
            hidden_layers = config["model"].get("parkinsons_hidden_layers", [128, 64, 32])
        elif dataset_type.lower() == "third_dataset":
            hidden_layers = config["model"].get("third_dataset_hidden_layers", [256, 128, 64])
        
        # Create model with the right architecture for the dataset
        model = create_model(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            dataset_type=dataset_type,
            task=task
        )
        logger.info(f"Created {dataset_type} model with input dim={input_dim}, output dim={output_dim}, task={task}")
    
    # Create client
    client = Client(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config["training"]["epochs"],
        client_id=client_id,
        server_host=server_host,
        server_port=server_port,
        dataset_type=dataset_type
    )
    
    # Start continuous learning
    client.start_continuous_learning(num_cycles=num_cycles, wait_time=wait_time)


if __name__ == "__main__":
    # Example usage
    import argparse
    import json
    from utils.data_utils import load_datasets
    
    parser = argparse.ArgumentParser(description="Run a federated learning client")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--client-id", type=str, default=None, help="Client identifier")
    parser.add_argument("--server-host", type=str, default="127.0.0.1", help="Server hostname or IP")
    parser.add_argument("--server-port", type=int, default=8080, help="Server port")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles (0 for infinite)")
    parser.add_argument("--wait-time", type=int, default=10, help="Wait time between cycles in seconds")
    parser.add_argument("--dataset-type", type=str, default="breast_cancer", 
                        choices=["breast_cancer", "parkinsons", "third_dataset"],
                        help="Type of dataset to use")
    parser.add_argument("--task", type=str, default=None,
                        choices=["classification", "regression"],
                        help="Task type (defaults based on dataset)")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Load dataset and create model
    dataset_path = config["dataset"].get("path")
    
    # Use dataset-specific path if available
    if args.dataset_type == "parkinsons" and "parkinsons_path" in config["dataset"]:
        dataset_path = config["dataset"]["parkinsons_path"]
    elif args.dataset_type == "third_dataset" and "third_dataset_path" in config["dataset"]:
        dataset_path = config["dataset"]["third_dataset_path"]
    
    target_column = config["dataset"].get("target_column")
    
    # Use dataset-specific target column if available
    if args.dataset_type == "parkinsons" and "parkinsons_target_column" in config["dataset"]:
        target_column = config["dataset"]["parkinsons_target_column"]
    elif args.dataset_type == "third_dataset" and "third_dataset_target_column" in config["dataset"]:
        target_column = config["dataset"]["third_dataset_target_column"]
    
    train_dataset, test_dataset, train_loader, test_loader = load_datasets(
        dataset_path,
        target_column,
        config["training"]["batch_size"],
        dataset_type=args.dataset_type
    )
    
    # Run client with specified dataset type
    run_client(
        config=config,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        test_loader=test_loader,
        client_id=args.client_id,
        server_host=args.server_host,
        server_port=args.server_port,
        num_cycles=args.cycles,
        wait_time=args.wait_time,
        dataset_type=args.dataset_type
    )