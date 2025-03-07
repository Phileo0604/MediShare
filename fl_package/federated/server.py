import os
import json
import time
import socket
import threading
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Optional
import logging

from models.nn_models import get_model_path

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

class ModelServer:
    """
    A federated learning server that runs continuously, accepting client contributions
    and updating the global model as contributions are received.
    """
    
    def __init__(
        self, 
        model_path: str = "global_model.json",
        model_config: Dict = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        update_threshold: int = 1,  # Number of clients needed before updating model
        contribution_weight: float = 0.1  # Weight for new contributions (0-1)
    ):
        self.model_path = model_path
        self.model_config = model_config
        self.host = host
        self.port = port
        self.update_threshold = update_threshold
        self.contribution_weight = contribution_weight
        
        # Initialize model parameters and tracking
        # Use dictionary to store multiple models based on dataset type
        self.global_parameters = {}
        self.pending_contributions = {}
        self.client_count = {}
        self.update_count = {}
        
        # Thread control
        self.running = False
        self.server_thread = None
        self.lock = threading.Lock()
        
        # Make sure the global_models directory exists
        os.makedirs("global_models", exist_ok=True)
        
        # Load existing models if available
        self.load_global_models()
        
    def load_global_models(self) -> None:
        """Load all global models from disk if they exist."""
        # List of supported dataset types
        dataset_types = ["breast_cancer", "parkinsons", "third_dataset"]
        
        for dataset_type in dataset_types:
            model_path = get_model_path(dataset_type)
            self.load_global_model(dataset_type, model_path)
    
    def load_global_model(self, dataset_type: str, model_path: str) -> None:
        """Load a global model for a specific dataset type if it exists."""
        if os.path.exists(model_path):
            try:
                # Check file extension to determine format
                if model_path.endswith('.pkl'):
                    # Pickle format for XGBoost models
                    with open(model_path, "rb") as f:
                        parameters = pickle.load(f)
                else:
                    # JSON format for neural network models
                    with open(model_path, "r") as f:
                        parameters_json = json.load(f)
                        parameters = [np.array(param) for param in parameters_json]
                
                self.global_parameters[dataset_type] = parameters
                logger.info(f"Loaded global model for {dataset_type} from {model_path}")
                
                # Initialize tracking for this dataset type
                if dataset_type not in self.pending_contributions:
                    self.pending_contributions[dataset_type] = []
                if dataset_type not in self.client_count:
                    self.client_count[dataset_type] = 0
                if dataset_type not in self.update_count:
                    self.update_count[dataset_type] = 0
                
            except Exception as e:
                logger.error(f"Error loading model for {dataset_type}: {e}")
                self.global_parameters[dataset_type] = None
        else:
            logger.warning(f"No model file found for {dataset_type} at {model_path}")
            self.global_parameters[dataset_type] = None
            
            # Initialize tracking for this dataset type
            self.pending_contributions[dataset_type] = []
            self.client_count[dataset_type] = 0
            self.update_count[dataset_type] = 0
    
    def save_global_model(self, dataset_type: str) -> None:
        """Save the global model for a specific dataset type."""
        if dataset_type in self.global_parameters and self.global_parameters[dataset_type] is not None:
            model_path = get_model_path(dataset_type)
            
            try:
                # Check if this is an XGBoost model (binary data in first parameter)
                is_xgboost = (len(self.global_parameters[dataset_type]) == 1 and 
                              isinstance(self.global_parameters[dataset_type][0], np.ndarray) and
                              self.global_parameters[dataset_type][0].dtype == np.uint8)
                
                if is_xgboost or model_path.endswith('.pkl'):
                    # Pickle format for XGBoost models
                    with open(model_path, "wb") as f:
                        pickle.dump(self.global_parameters[dataset_type], f)
                else:
                    # JSON format for neural network models
                    parameters_list = [param.tolist() for param in self.global_parameters[dataset_type]]
                    with open(model_path, "w") as f:
                        json.dump(parameters_list, f)
                
                # Create a timestamped backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = os.path.join("model_backups", dataset_type)
                os.makedirs(backup_dir, exist_ok=True)
                
                if model_path.endswith('.json'):
                    backup_path = os.path.join(backup_dir, f"global_model_{timestamp}.json")
                else:
                    backup_path = os.path.join(backup_dir, f"global_model_{timestamp}.pkl")
                
                # Save backup using same format as original
                if is_xgboost or model_path.endswith('.pkl'):
                    with open(backup_path, "wb") as f:
                        pickle.dump(self.global_parameters[dataset_type], f)
                else:
                    with open(backup_path, "w") as f:
                        json.dump(parameters_list, f)
                
                logger.info(f"Saved global model for {dataset_type} to {model_path} with backup at {backup_path}")
            except Exception as e:
                logger.error(f"Error saving model for {dataset_type}: {e}")
    
    def update_global_model(self, client_parameters: List[np.ndarray], dataset_type: str) -> None:
        """
        Update the global model with client parameters for a specific dataset type.
        Uses weighted averaging between current global model and client contribution.
        """
        # No need to acquire lock here since this is called from within a locked context in process_client_contribution
        
        # If global parameters don't exist yet for this dataset type, use client parameters directly
        if dataset_type not in self.global_parameters or self.global_parameters[dataset_type] is None:
            self.global_parameters[dataset_type] = [np.copy(param) for param in client_parameters]
            logger.info(f"Initialized global model for {dataset_type} with first client contribution")
        else:
            # Check if this is an XGBoost model (binary data in first parameter)
            is_xgboost = (len(self.global_parameters[dataset_type]) == 1 and 
                          isinstance(self.global_parameters[dataset_type][0], np.ndarray) and
                          self.global_parameters[dataset_type][0].dtype == np.uint8)
            
            if is_xgboost:
                # For XGBoost models, we just replace the model with the latest contribution
                # This is because averaging binary XGBoost models is not straightforward
                logger.info(f"Updating XGBoost model for {dataset_type} with latest contribution")
                self.global_parameters[dataset_type] = [np.copy(param) for param in client_parameters]
            else:
                # For neural network models, perform weighted averaging
                # Validate that shapes match
                if len(self.global_parameters[dataset_type]) != len(client_parameters):
                    logger.error(f"Parameter shape mismatch for {dataset_type}: global {len(self.global_parameters[dataset_type])} vs client {len(client_parameters)}")
                    return
                
                # Update global model with weighted average
                for i in range(len(self.global_parameters[dataset_type])):
                    if self.global_parameters[dataset_type][i].shape != client_parameters[i].shape:
                        logger.error(f"Parameter shape mismatch for {dataset_type} at index {i}")
                        return
                    
                    # Weighted average: (1-w)*global + w*client
                    self.global_parameters[dataset_type][i] = (1 - self.contribution_weight) * self.global_parameters[dataset_type][i] + \
                                                             self.contribution_weight * client_parameters[i]
        
        # Increment update counter for this dataset type
        if dataset_type not in self.update_count:
            self.update_count[dataset_type] = 0
        self.update_count[dataset_type] += 1
        
        logger.info(f"Updated global model for {dataset_type} (update #{self.update_count[dataset_type]})")
        
        # Save the updated model
        self.save_global_model(dataset_type)
    
    def process_client_contribution(self, client_parameters: List[np.ndarray], dataset_type: str) -> None:
        """Process a client contribution for a specific dataset type."""
        with self.lock:
            # Initialize tracking for this dataset type if needed
            if dataset_type not in self.pending_contributions:
                self.pending_contributions[dataset_type] = []
            if dataset_type not in self.client_count:
                self.client_count[dataset_type] = 0
                
            # Add contribution to pending list
            self.pending_contributions[dataset_type].append(client_parameters)
            self.client_count[dataset_type] += 1
            
            logger.info(f"Received client contribution for {dataset_type} (client #{self.client_count[dataset_type]}, pending: {len(self.pending_contributions[dataset_type])})")
            
            # Check if we should update the global model
            if len(self.pending_contributions[dataset_type]) >= self.update_threshold:
                logger.info(f"Update threshold reached for {dataset_type} ({self.update_threshold}), updating global model")
                # Simple approach: use the latest contribution
                self.update_global_model(self.pending_contributions[dataset_type][-1], dataset_type)
                
                # Alternative: average all pending contributions
                # avg_parameters = self._average_parameters(self.pending_contributions[dataset_type])
                # self.update_global_model(avg_parameters, dataset_type)
                
                # Clear pending contributions after update
                self.pending_contributions[dataset_type] = []
                logger.info(f"Cleared pending contributions for {dataset_type} after update")
    
    def _average_parameters(self, parameter_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Average multiple sets of parameters."""
        # Initialize with first client's parameters
        avg_params = [np.zeros_like(param) for param in parameter_list[0]]
        
        # Sum parameters from all clients
        for parameters in parameter_list:
            for i, param in enumerate(parameters):
                avg_params[i] += param
        
        # Divide by number of clients to get average
        n_clients = len(parameter_list)
        for i in range(len(avg_params)):
            avg_params[i] /= n_clients
            
        return avg_params
    
    def handle_client(self, client_socket: socket.socket, client_address: tuple) -> None:
        """Handle a client connection."""
        try:
            logger.info(f"Client connected from {client_address}")
            
            # First receive the dataset type from the client
            size_bytes = client_socket.recv(8)
            if not size_bytes:
                logger.warning(f"Client {client_address} disconnected before sending dataset type size")
                return
                
            data_size = int.from_bytes(size_bytes, byteorder='big')
            
            # Receive dataset type
            dataset_type_bytes = b''
            while len(dataset_type_bytes) < data_size:
                chunk = client_socket.recv(min(4096, data_size - len(dataset_type_bytes)))
                if not chunk:
                    break
                dataset_type_bytes += chunk
                
            if len(dataset_type_bytes) == data_size:
                dataset_type = dataset_type_bytes.decode('utf-8')
                logger.info(f"Client {client_address} requested model for dataset: {dataset_type}")
            else:
                logger.warning(f"Incomplete dataset type from client {client_address}")
                return
            
            # Send current global model to client if available for the requested dataset type
            if dataset_type in self.global_parameters and self.global_parameters[dataset_type] is not None:
                # Check if this is an XGBoost model (binary data needs to be serialized differently)
                is_xgboost = (len(self.global_parameters[dataset_type]) == 1 and 
                              isinstance(self.global_parameters[dataset_type][0], np.ndarray) and
                              self.global_parameters[dataset_type][0].dtype == np.uint8)
                
                if is_xgboost:
                    # Use pickle for XGBoost models
                    parameters_bytes = pickle.dumps(self.global_parameters[dataset_type])
                else:
                    # Use JSON for neural network models
                    parameters_json = json.dumps([param.tolist() for param in self.global_parameters[dataset_type]])
                    parameters_bytes = parameters_json.encode('utf-8')
                
                # Send the size of the data first
                size_bytes = len(parameters_bytes).to_bytes(8, byteorder='big')
                client_socket.sendall(size_bytes)
                
                # Send the parameters
                client_socket.sendall(parameters_bytes)
                logger.info(f"Sent global model for {dataset_type} to client {client_address}")
            else:
                # Send empty model indicator
                client_socket.sendall((0).to_bytes(8, byteorder='big'))
                logger.info(f"Sent empty model indicator for {dataset_type} to client {client_address}")
            
            # Receive updated parameters from client
            # First get the size of incoming data
            size_bytes = client_socket.recv(8)
            if not size_bytes:
                logger.warning(f"Client {client_address} disconnected before sending parameters")
                return
                
            data_size = int.from_bytes(size_bytes, byteorder='big')
            logger.info(f"Expecting {data_size} bytes from client {client_address}")
            
            # Receive all data
            received_data = b''
            while len(received_data) < data_size:
                chunk = client_socket.recv(min(4096, data_size - len(received_data)))
                if not chunk:
                    logger.warning(f"Client {client_address} disconnected during data transfer")
                    break
                received_data += chunk
            
            # Now check if we received complete data
            if len(received_data) == data_size:
                logger.info(f"Successfully received complete data ({len(received_data)} bytes) from client {client_address}")
                
                try:
                    # Check if this is likely to be pickle data (for XGBoost)
                    is_xgboost = (get_model_path(dataset_type).endswith('.pkl') or 
                                 (dataset_type.lower() in ["parkinsons", "third_dataset"]))
                    
                    if is_xgboost:
                        # Parse as pickle
                        client_parameters = pickle.loads(received_data)
                    else:
                        # Parse as JSON
                        parameters_json = received_data.decode('utf-8')
                        client_parameters = [np.array(param) for param in json.loads(parameters_json)]
                    
                    logger.info(f"Successfully parsed parameters for {dataset_type} from client {client_address}")
                    
                    # Process the contribution for the specific dataset type
                    self.process_client_contribution(client_parameters, dataset_type)
                    
                    # Send acknowledgment
                    client_socket.sendall(b'ACK')
                    logger.info(f"Processed contribution for {dataset_type} from client {client_address}")
                except Exception as e:
                    logger.error(f"Error processing parameters from client {client_address}: {e}")
            else:
                logger.warning(f"Incomplete data from client {client_address}: expected {data_size} bytes, received {len(received_data)} bytes")
        
        except Exception as e:
            logger.error(f"Error handling client {client_address}: {e}")
        finally:
            client_socket.close()
    
    def start_server(self) -> None:
        """Start the server and listen for connections."""
        self.running = True
        
        def server_thread_func():
            try:
                # Create server socket
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind((self.host, self.port))
                server_socket.listen(10)  # Allow up to 10 queued connections
                
                logger.info(f"Server listening on {self.host}:{self.port}")
                
                while self.running:
                    try:
                        # Accept client connection
                        client_socket, client_address = server_socket.accept()
                        
                        # Handle client in a separate thread
                        client_thread = threading.Thread(
                            target=self.handle_client,
                            args=(client_socket, client_address)
                        )
                        client_thread.daemon = True
                        client_thread.start()
                    except Exception as e:
                        if self.running:  # Only log if we're still supposed to be running
                            logger.error(f"Error accepting client: {e}")
            
            except Exception as e:
                logger.error(f"Server error: {e}")
            finally:
                if 'server_socket' in locals():
                    server_socket.close()
                logger.info("Server stopped")
        
        # Start server in a separate thread
        self.server_thread = threading.Thread(target=server_thread_func)
        self.server_thread.daemon = True
        self.server_thread.start()
    
    def stop_server(self) -> None:
        """Stop the server."""
        self.running = False
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5.0)
        logger.info("Server shutdown complete")


def run_server(config):
    """Run the federated learning server."""
    # Extract server configuration
    host = config["server"].get("host", "0.0.0.0")
    port = int(config["server"].get("port", 8080))
    update_threshold = config["server"].get("update_threshold", 1)
    contribution_weight = config["server"].get("contribution_weight", 0.1)
    
    # Create model configuration
    model_config = {
        "input_dim": config.get("model_input_dim", 30),
        "output_dim": config.get("model_output_dim", 2),
        "hidden_layers": config["model"].get("hidden_layers", [64, 32])
    }
    
    # Create and start server
    server = ModelServer(
        model_path=config["model"].get("parameters_file", "global_model.json"),
        model_config=model_config,
        host=host,
        port=port,
        update_threshold=update_threshold,
        contribution_weight=contribution_weight
    )
    
    # Start the server
    server.start_server()
    
    logger.info("Server started. Press Ctrl+C to stop...")
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop_server()


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