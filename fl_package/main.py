from flwr.simulation import run_simulation
from federated import client_fn, server_fn
from flwr.server import ServerApp
from flwr.client import ClientApp
import json


with open("config.json", "r") as f:
    config = json.load(f)

NUM_CLIENTS = config["NUM_CLIENTS"]

server = ServerApp(server_fn=server_fn)
client = ClientApp(client_fn=client_fn)

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS
)
