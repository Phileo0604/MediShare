import socket
import json
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional
import time

from models.nn_models import get_parameters, set_parameters
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
        retry_interval: int = 10  # Seconds to wait between retries
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.client_id = client_id or "unknown"
        self.server_host = server_host
        self.server_port = server_port
        self.retry_interval = retry_interval
    
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
            # First get the size of incoming data
            size_bytes = client_socket.recv(8)
            if not size_bytes:
                logger.warning("Server disconnected before sending model size")
                return False
                
            data_size = int.from_bytes(size_bytes, byteorder='big')
            
            # If data_size is 0, server has no model yet
            if data_size == 0:
                logger.info("Server has no global model yet")
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
                parameters_json = received_data.decode('utf-8')
                global_parameters = [np.array(param) for param in json.loads(parameters_json)]
                
                # Update model with global parameters
                set_parameters(self.model, global_parameters)
                logger.info(f"Received and set global model parameters")
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
            
            # Serialize parameters to JSON
            parameters_json = json.dumps([param.tolist() for param in parameters])
            parameters_bytes = parameters_json.encode('utf-8')
            
            # Send the size of the data first
            size_bytes = len(parameters_bytes).to_bytes(8, byteorder='big')
            client_socket.sendall(size_bytes)
            
            # Send the parameters
            client_socket.sendall(parameters_bytes)
            logger.info(f"Sent updated model parameters ({len(parameters_bytes)} bytes)")
            
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
        logger.info(f"Starting local training for {self.epochs} epochs")
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
    
    def start_continuous_learning(self, num_cycles: int = 0, wait_time: int = 60) -> None:
        """
        Start continuous federated learning process.
        
        Args:
            num_cycles: Number of cycles to run (0 for infinite)
            wait_time: Time to wait between cycles in seconds
        """
        cycle_count = 0
        
        while num_cycles == 0 or cycle_count < num_cycles:
            cycle_count += 1
            logger.info(f"Starting federated cycle {cycle_count}")
            
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
    model,
    client_id=None,
    server_host="127.0.0.1",
    server_port=8080,
    num_cycles=0,
    wait_time=60
):
    """Run a continuous federated learning client."""
    # Create client
    client = Client(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config["training"]["epochs"],
        client_id=client_id,
        server_host=server_host,
        server_port=server_port
    )
    
    # Start continuous learning
    client.start_continuous_learning(num_cycles=num_cycles, wait_time=wait_time)


if __name__ == "__main__":
    # Example usage
    import argparse
    import json
    from utils.data_utils import load_datasets
    from models.nn_models import create_model
    
    parser = argparse.ArgumentParser(description="Run a federated learning client")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--client-id", type=str, default=None, help="Client identifier")
    parser.add_argument("--server-host", type=str, default="127.0.0.1", help="Server hostname or IP")
    parser.add_argument("--server-port", type=int, default=8080, help="Server port")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles (0 for infinite)")
    parser.add_argument("--wait-time", type=int, default=60, help="Wait time between cycles in seconds")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Load dataset and create model
    train_dataset, test_dataset, train_loader, test_loader = load_datasets(
        config["dataset"]["path"],
        config["dataset"]["target_column"],
        config["training"]["batch_size"]
    )
    
    # Get model dimensions from dataset
    input_dim = train_dataset.features.shape[1]
    output_dim = len(set(train_dataset.labels))
    
    # Create model
    model = create_model(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=config["model"]["hidden_layers"]
    )
    
    # Run client
    run_client(
        config=config,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        client_id=args.client_id,
        server_host=args.server_host,
        server_port=args.server_port,
        num_cycles=args.cycles,
        wait_time=args.wait_time
    )