import os
import json
import time
import socket
import threading
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Optional
import logging

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
        self.global_parameters = None
        self.pending_contributions = []
        self.client_count = 0
        self.update_count = 0
        
        # Thread control
        self.running = False
        self.server_thread = None
        self.lock = threading.Lock()
        
        # Load existing model if available
        self.load_global_model()
        
    def load_global_model(self) -> None:
        """Load the global model from disk if it exists."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r") as f:
                    parameters = json.load(f)
                
                # Convert to numpy arrays
                self.global_parameters = [np.array(param) for param in parameters]
                logger.info(f"Loaded global model from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.global_parameters = None
        else:
            logger.warning(f"No model file found at {self.model_path}")
            self.global_parameters = None
    
    def save_global_model(self) -> None:
        print("DEBUG: Entered save_global_model method")
        """Save the global model to disk."""
        if self.global_parameters is not None:
            try:
                # Convert parameters to list for JSON
                parameters_list = [param.tolist() for param in self.global_parameters]
                
                # Save parameters
                with open(self.model_path, "w") as f:
                    json.dump(parameters_list, f)
                
                # Create a timestamped backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = "model_backups"
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, f"global_model_{timestamp}.json")
                
                with open(backup_path, "w") as f:
                    json.dump(parameters_list, f)
                
                logger.info(f"Saved global model to {self.model_path} with backup at {backup_path}")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
    
    def update_global_model(self, client_parameters: List[np.ndarray]) -> None:
        """
        Update the global model with client parameters.
        Uses weighted averaging between current global model and client contribution.
        """
        with self.lock:
            # If global parameters don't exist yet, use client parameters directly
            if self.global_parameters is None:
                self.global_parameters = [np.copy(param) for param in client_parameters]
                logger.info("Initialized global model with first client contribution")
            else:
                # Validate that shapes match
                if len(self.global_parameters) != len(client_parameters):
                    logger.error(f"Parameter shape mismatch: global {len(self.global_parameters)} vs client {len(client_parameters)}")
                    return
                
                # Update global model with weighted average
                for i in range(len(self.global_parameters)):
                    if self.global_parameters[i].shape != client_parameters[i].shape:
                        logger.error(f"Parameter shape mismatch at index {i}")
                        return
                    
                    # Weighted average: (1-w)*global + w*client
                    self.global_parameters[i] = (1 - self.contribution_weight) * self.global_parameters[i] + \
                                               self.contribution_weight * client_parameters[i]
            
            self.update_count += 1
            logger.info(f"Updated global model (update #{self.update_count})")
            
            # Save the updated model
            print(f"DEBUG: About to save global model, update count: {self.update_count}")
            self.save_global_model()
    
    def process_client_contribution(self, client_parameters: List[np.ndarray]) -> None:
        """Process a client contribution by updating the pending contributions list."""
        with self.lock:
            self.pending_contributions.append(client_parameters)
            self.client_count += 1
            
            logger.info(f"Received client contribution (client #{self.client_count}, pending: {len(self.pending_contributions)})")
            
            # Check if we should update the global model
            if len(self.pending_contributions) >= self.update_threshold:
                logger.info(f"Update threshold reached ({self.update_threshold}), updating global model")
                # Simple approach: use the latest contribution
                self.update_global_model(self.pending_contributions[-1])
                
                # Alternative: average all pending contributions
                # avg_parameters = self._average_parameters(self.pending_contributions)
                # self.update_global_model(avg_parameters)
                
                # Clear pending contributions after update
                self.pending_contributions = []
                logger.info("Cleared pending contributions after update")
    
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
            
            # Send current global model to client if available
            if self.global_parameters is not None:
                # Serialize parameters to JSON
                parameters_json = json.dumps([param.tolist() for param in self.global_parameters])
                
                # Send the size of the data first
                size_bytes = len(parameters_json).to_bytes(8, byteorder='big')
                client_socket.sendall(size_bytes)
                
                # Send the parameters
                client_socket.sendall(parameters_json.encode('utf-8'))
                logger.info(f"Sent global model to client {client_address}")
            else:
                # Send empty model indicator
                client_socket.sendall((0).to_bytes(8, byteorder='big'))
                logger.info(f"Sent empty model indicator to client {client_address}")
            
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
                    parameters_json = received_data.decode('utf-8')
                    client_parameters = [np.array(param) for param in json.loads(parameters_json)]
                    logger.info(f"Successfully parsed parameters from client {client_address}")
                    
                    # Process the contribution
                    self.process_client_contribution(client_parameters)
                    
                    # Send acknowledgment
                    client_socket.sendall(b'ACK')
                    logger.info(f"Processed contribution from client {client_address}")
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
    model_path = config["model"].get("parameters_file", "global_model.json")
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
        model_path=model_path,
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