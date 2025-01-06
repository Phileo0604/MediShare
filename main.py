import json

from federated_learning.federated_client import FederatedClient

if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    
    client = FederatedClient(config)
    client.train(config["epochs"])
    parameters = client.get_parameters()

    # Send parameters to server (implementation dependent)
    print("Trained model parameters sent to the server.")
