import socket
import threading
import json
import numpy as np
import torch
import pickle
import logging
import time
import os
from typing import Dict, List, Any, Optional

from models.nn_models import create_model, get_parameters, set_parameters, export_model_parameters
from utils.server_utils import backup_global_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Server")

class Server:
    """
    Federated learning server that aggregates model updates from clients.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 host: str = "0.0.0.0", 
                 port: int = 8080,
                 update_threshold: int = 1,
                 contribution_weight: float = 0.1):
        self.host = host
        self.port = port
        self.update_threshold = update_threshold
        self.contribution_weight = contribution_weight
        self.config = config
        self.running = False
        self.server_socket = None
        
        # Initialize models registry to store models for each dataset type
        self.models_registry = {}
        self.model_parameters_registry = {}
        self.client_updates_registry = {}
        self.update_counts_registry = {}
        
        # Extract supported datasets from config
        self.supported_datasets = config.get("supported_datasets", ["breast_cancer", "parkinsons", "reinopath"])
        logger.info(f"Server initialized with support for dataset types: {', '.join(self.supported_datasets)}")
        
        # Initialize models for all supported datasets
        for dataset_type in self.supported_datasets:
            self.initialize_model_for_dataset_type(dataset_type)
    
    def initialize_model_for_dataset_type(self, dataset_type: str) -> None:
        """Initialize a model for a specific dataset type."""
        try:
            logger.info(f"Initializing model for {dataset_type}")
            
            # Get dataset-specific configuration
            dataset_configs = self.config.get("dataset_configs", {})
            dataset_config = dataset_configs.get(dataset_type, {})
            
            # Determine model architecture parameters based on dataset type
            if dataset_type == "breast_cancer":
                input_dim = 30
                output_dim = 2
                task = "classification"
                # Standard neural network model
                model_type = "nn"
            elif dataset_type == "parkinsons":
                input_dim = 16
                output_dim = 1
                task = "regression"
                # XGBoost model - needs special handling
                model_type = "xgboost"
            elif dataset_type == "reinopath":
                input_dim = 19
                output_dim = 2
                task = "classification"
                # XGBoost model - needs special handling
                model_type = "xgboost"
            else:
                # Default values if unknown dataset type
                input_dim = 10
                output_dim = 2
                task = "classification"
                model_type = "nn"
            
            # Get hidden layers configuration
            hidden_layers = dataset_config.get("hidden_layers", [64, 32])
            
            # Check if this is an XGBoost model type that needs special handling
            if model_type == "xgboost":
                # For XGBoost models, we'll create a placeholder and defer actual initialization
                logger.info(f"Deferring full initialization of {dataset_type} model until client data is available")
                
                # Store basic model properties instead of the actual model
                self.models_registry[dataset_type] = {
                    "type": model_type,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "hidden_layers": hidden_layers,
                    "task": task,
                    "initialized": False
                }
                
                # Set empty parameters or load from file if it exists
                parameters_file = dataset_config.get("parameters_file")
                
                if parameters_file and os.path.exists(parameters_file):
                    try:
                        # Try to load parameters
                        with open(parameters_file, "rb") as f:
                            if parameters_file.endswith(".pkl"):
                                self.model_parameters_registry[dataset_type] = pickle.load(f)
                            else:
                                # JSON format
                                json_data = json.load(f)
                                self.model_parameters_registry[dataset_type] = [np.array(param) for param in json_data]
                        logger.info(f"Loaded parameters for {dataset_type} from {parameters_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load parameters for {dataset_type}, using empty placeholder: {e}")
                        # Use an empty list as placeholder for parameters
                        self.model_parameters_registry[dataset_type] = []
                else:
                    # Use an empty list as placeholder for parameters
                    self.model_parameters_registry[dataset_type] = []
                    
            else:
                # Standard neural network model - create normally
                model = create_model(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_layers=hidden_layers,
                    dataset_type=dataset_type,
                    task=task
                )
                
                # Check if we have a parameters file to load
                parameters_file = dataset_config.get("parameters_file") 
                if parameters_file and os.path.exists(parameters_file):
                    try:
                        from models.nn_models import import_model_parameters
                        import_model_parameters(model, parameters_file)
                        logger.info(f"Loaded parameters for {dataset_type} from {parameters_file}")
                    except Exception as e:
                        logger.error(f"Failed to load parameters for {dataset_type}: {e}")
                
                # Store the model and get parameters
                self.models_registry[dataset_type] = model
                self.model_parameters_registry[dataset_type] = get_parameters(model)
            
            # Initialize update tracking for this dataset type
            self.client_updates_registry[dataset_type] = []
            self.update_counts_registry[dataset_type] = 0
            
            logger.info(f"Model for {dataset_type} initialized")
        except Exception as e:
            logger.error(f"Error initializing model for {dataset_type}: {e}")
    
    def start(self) -> None:
        """Start the server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        self.running = True
        logger.info(f"Server started on {self.host}:{self.port}")
        
        try:
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    logger.info(f"New connection from {address}")
                    
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except Exception as e:
                    logger.error(f"Error accepting client connection: {e}")
        except KeyboardInterrupt:
            logger.info("Server stopping due to keyboard interrupt")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("Server stopped")
    
    def handle_client(self, client_socket: socket.socket, address: tuple) -> None:
        """Handle communication with a client."""
        try:
            # First receive the dataset type from the client
            size_bytes = client_socket.recv(8)
            if not size_bytes:
                logger.warning(f"Client {address} disconnected before sending dataset type size")
                client_socket.close()
                return
                
            data_size = int.from_bytes(size_bytes, byteorder='big')
            
            # Receive the dataset type
            dataset_type_bytes = client_socket.recv(data_size)
            if len(dataset_type_bytes) < data_size:
                logger.warning(f"Incomplete dataset type data from client {address}")
                client_socket.close()
                return
                
            dataset_type = dataset_type_bytes.decode('utf-8')
            logger.info(f"Client {address} requested model for dataset type: {dataset_type}")
            
            # Check if we support this dataset type
            if dataset_type not in self.models_registry:
                logger.warning(f"Unsupported dataset type from client {address}: {dataset_type}")
                # Send empty response (size 0)
                client_socket.sendall((0).to_bytes(8, byteorder='big'))
                client_socket.close()
                return
            
            # Send current global model
            self.send_global_model(client_socket, dataset_type)
            
            # Receive client's updated model
            if self.receive_client_update(client_socket, dataset_type):
                # Check if we've reached threshold for this dataset type
                if len(self.client_updates_registry[dataset_type]) >= self.update_threshold:
                    self.aggregate_updates(dataset_type)
                    # Create a backup
                    self.backup_model(dataset_type)
            
        except Exception as e:
            logger.error(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
    
    def send_global_model(self, client_socket: socket.socket, dataset_type: str) -> None:
        """Send the current global model for the specified dataset type to a client."""
        try:
            parameters = self.model_parameters_registry[dataset_type]
            
            # Determine if this is an XGBoost model by checking the model registry
            is_xgboost = isinstance(self.models_registry[dataset_type], dict) and self.models_registry[dataset_type].get("type") == "xgboost"
            
            if is_xgboost:
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
            logger.info(f"Sent global model for {dataset_type} ({len(parameters_bytes)} bytes)")
        
        except Exception as e:
            logger.error(f"Error sending global model: {e}")
            # Send empty response to indicate error
            client_socket.sendall((0).to_bytes(8, byteorder='big'))
    
    def receive_client_update(self, client_socket: socket.socket, dataset_type: str) -> bool:
        """Receive a model update from a client."""
        try:
            # First get the size of incoming data
            size_bytes = client_socket.recv(8)
            if not size_bytes:
                logger.warning("Client disconnected before sending update size")
                return False
                
            data_size = int.from_bytes(size_bytes, byteorder='big')
            logger.info(f"Expecting {data_size} bytes from client for {dataset_type}")
            
            # Receive all data
            received_data = b''
            while len(received_data) < data_size:
                chunk = client_socket.recv(min(4096, data_size - len(received_data)))
                if not chunk:
                    break
                received_data += chunk
            
            # Parse received parameters
            if len(received_data) == data_size:
                # Determine if this is an XGBoost model
                is_xgboost = isinstance(self.models_registry[dataset_type], dict) and self.models_registry[dataset_type].get("type") == "xgboost"
                
                if is_xgboost:
                    # Parse as pickle for XGBoost
                    client_parameters = pickle.loads(received_data)
                    
                    # If this is the first update and model isn't initialized yet, update model info
                    if not self.models_registry[dataset_type].get("initialized", False):
                        self.models_registry[dataset_type]["initialized"] = True
                        logger.info(f"XGBoost model for {dataset_type} is now initialized with client data")
                else:
                    # Parse as JSON for neural networks
                    parameters_json = received_data.decode('utf-8')
                    client_parameters = [np.array(param) for param in json.loads(parameters_json)]
                
                # Add to client updates for this dataset type
                self.client_updates_registry[dataset_type].append(client_parameters)
                logger.info(f"Received client update for {dataset_type} (client count: {len(self.client_updates_registry[dataset_type])})")
                
                # Send acknowledgment
                client_socket.sendall(b'ACK')
                return True
            else:
                logger.warning(f"Incomplete data from client")
                return False
        
        except Exception as e:
            logger.error(f"Error receiving client update: {e}")
            return False
    
    def aggregate_updates(self, dataset_type: str) -> None:
        """Aggregate client updates for the specified dataset type and update the global model."""
        try:
            # Get current global parameters for this dataset type
            global_parameters = self.model_parameters_registry[dataset_type]
            client_updates = self.client_updates_registry[dataset_type]
            
            if not client_updates:
                logger.warning(f"No client updates to aggregate for {dataset_type}")
                return
            
            logger.info(f"Aggregating {len(client_updates)} client updates for {dataset_type}")
            
            # Check if this is an XGBoost model or neural network
            is_xgboost = isinstance(self.models_registry[dataset_type], dict) and self.models_registry[dataset_type].get("type") == "xgboost"
            
            if is_xgboost:
                # For XGBoost models, we're storing parameter bytes directly
                # Just use the most recent client update for now, could implement averaging later
                # Alternatively, you could keep global_parameters as is, applying the contribution weight
                if client_updates:
                    # Simple approach: just use the most recent update
                    global_parameters = client_updates[-1]
            else:
                # Neural network aggregation logic
                # Compute the average of client updates
                for client_parameters in client_updates:
                    for i, param in enumerate(client_parameters):
                        # Scale the client's contribution
                        contribution = self.contribution_weight * param
                        
                        # Update the global parameters
                        global_parameters[i] = global_parameters[i] * (1 - self.contribution_weight) + contribution
                
                # Update the actual model with new parameters if it's a neural network
                set_parameters(self.models_registry[dataset_type], global_parameters)
            
            # Update the global parameters in our registry
            self.model_parameters_registry[dataset_type] = global_parameters
            
            # Increment update count for this dataset type
            self.update_counts_registry[dataset_type] += 1
            
            # Clear client updates for this dataset type
            self.client_updates_registry[dataset_type] = []
            
            logger.info(f"Global model for {dataset_type} updated (update #{self.update_counts_registry[dataset_type]})")
        
        except Exception as e:
            logger.error(f"Error aggregating updates for {dataset_type}: {e}")
    
    def backup_model(self, dataset_type: str) -> None:
        """Create a backup of the global model for the specified dataset type."""
        try:
            # Get backup directory from config for this dataset type
            dataset_configs = self.config.get("dataset_configs", {})
            dataset_config = dataset_configs.get(dataset_type, {})
            
            backup_dir = dataset_config.get("backup_dir", f"model_backups/{dataset_type}")
            
            # Create backup directory if it doesn't exist
            os.makedirs(backup_dir, exist_ok=True)
            
            # Get the model's parameters file path
            parameters_file = dataset_config.get("parameters_file", f"global_models/{dataset_type}_model.json")
            
            # Determine if this is an XGBoost model
            is_xgboost = isinstance(self.models_registry[dataset_type], dict) and self.models_registry[dataset_type].get("type") == "xgboost"
            
            if is_xgboost:
                # Export directly to pickle file for XGBoost
                if parameters_file.endswith(".json"):
                    # Convert to .pkl extension
                    parameters_file = parameters_file.replace(".json", ".pkl")
                
                # Save the parameters directly to file
                with open(parameters_file, "wb") as f:
                    pickle.dump(self.model_parameters_registry[dataset_type], f)
                
                logger.info(f"Saved XGBoost parameters to {parameters_file}")
            else:
                # Export neural network parameters
                export_model_parameters(self.models_registry[dataset_type], parameters_file)
            
            # Create a timestamped backup
            timestamp = int(time.time())
            if parameters_file.endswith(".json"):
                backup_file = f"{backup_dir}/model_{dataset_type}_{timestamp}.json"
                # Copy the file
                import shutil
                shutil.copy2(parameters_file, backup_file)
            else:
                backup_file = f"{backup_dir}/model_{dataset_type}_{timestamp}.pkl"
                # Copy the file
                import shutil
                shutil.copy2(parameters_file, backup_file)
            
            logger.info(f"Created backup of {dataset_type} model at {backup_file}")
        
        except Exception as e:
            logger.error(f"Error creating backup for {dataset_type}: {e}")


def run_server(config):
    """Run the federated learning server."""
    # Extract server configuration
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8080)
    update_threshold = server_config.get("update_threshold", 1)
    contribution_weight = server_config.get("contribution_weight", 0.1)
    
    # Start the server
    server = Server(
        config=config,
        host=host,
        port=port,
        update_threshold=update_threshold,
        contribution_weight=contribution_weight
    )
    
    server.start()


if __name__ == "__main__":
    # Example usage
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Run a federated learning server")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = json.load(f)
    
    run_server(config)