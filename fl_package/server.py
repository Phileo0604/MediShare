import json
from flwr.server import start_server
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

NUM_ROUNDS = config["NUM_ROUNDS"]

if __name__ == "__main__":
    print("🚀 Starting Federated Learning Server...")
    start_server(
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=FedAvg()
    )
