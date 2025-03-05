import torch
import flwr as fl
from flwr.client import NumPyClient, Client
from typing import Dict, List, Tuple, Union, Optional
import numpy as np

# This is a compatibility layer for older versions of Flower
# It manually implements the Client wrapper for NumPyClient


class NumPyClientAdapter(Client):
    """Wrapper for NumPyClient that implements the Client interface.
    This is for compatibility with older versions of Flower.
    """

    def __init__(self, numpy_client: NumPyClient):
        self.numpy_client = numpy_client

    def get_parameters(self, ins: Dict[str, bytes]) -> List[np.ndarray]:
        """Return the current local model parameters."""
        return self.numpy_client.get_parameters(ins)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, bytes]) -> Tuple[List[np.ndarray], int, Dict[str, bytes]]:
        """Train the provided parameters using the locally held dataset."""
        return self.numpy_client.fit(parameters, config)

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, bytes]) -> Tuple[float, int, Dict[str, bytes]]:
        """Evaluate the provided parameters using the locally held dataset."""
        return self.numpy_client.evaluate(parameters, config)


def start_client_compat(server_address: str, client: NumPyClient):
    """Start a client using the appropriate method based on Flower version."""
    try:
        # Try approach for newer versions
        fl.client.start_numpy_client(server_address=server_address, client=client)
    except (TypeError, AttributeError):
        try:
            # Try approach for older versions with keyword args
            fl.client.start_client(server_address=server_address, client=NumPyClientAdapter(client))
        except (TypeError, AttributeError):
            # Try approach for older versions with positional args
            fl.client.start_client(server_address, NumPyClientAdapter(client))