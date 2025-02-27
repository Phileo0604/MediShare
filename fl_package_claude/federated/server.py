from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context


def create_server_app(config):
    """Create a Flower server application with the specified configuration."""
    
    def server_fn(context: Context):
        # Create federated averaging strategy
        strategy = FedAvg()
        
        # Create server configuration
        server_config = ServerConfig(num_rounds=config["federated"]["num_rounds"])
        
        # Return server components
        return ServerAppComponents(strategy=strategy, config=server_config)
    
    return ServerApp(server_fn=server_fn)