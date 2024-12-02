import flwr as fl
from flwr.server import ServerConfig

# Define the strategy for federated learning
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,  # Percentage of clients to train
    fraction_evaluate=0.1,  # Percentage of clients to evaluate
    min_fit_clients=2,  # Minimum number of clients for training
    min_evaluate_clients=2,  # Minimum number of clients for evaluation
    min_available_clients=2,  # Minimum available clients
)

# Start the server
fl.server.start_server(
    config=ServerConfig(num_rounds=5),
    strategy=strategy,
    )
