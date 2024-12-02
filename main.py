import json

from federated_learning.general_client import FederatedClient

if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)

    client = FederatedClient(config)
    client.train(config["epochs"])

    # Evaluate the model on test data
    accuracy, avg_loss = client.evaluate()

    # Send parameters to server (implementation dependent)
    parameters = client.get_parameters()
    print("Trained model parameters sent to the server.")
