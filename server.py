import flwr as fl
from flwr.server.strategy import FedAvg

# Define strategy
strategy = FedAvg()

# Start server
if __name__ == "__main__":
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy)
