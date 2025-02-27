from flwr.client import NumPyClient, Client, ClientApp
from flwr.common import Context

from models.nn_models import get_parameters, set_parameters, import_model_parameters
from training.trainer import train, test


class FlowerCSVClient(NumPyClient):
    """Flower client for federated learning with CSV datasets."""
    
    def __init__(self, model, train_loader, test_loader, epochs):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays."""
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        set_parameters(self.model, parameters)
        train(self.model, self.train_loader, epochs=self.epochs)
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test dataset."""
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}


def create_client_app(config, train_dataset, test_dataset, train_loader, test_loader, device):
    """Create a Flower client application."""
    from models.nn_models import create_model

    def client_fn(context: Context):
        # Get model dimensions from dataset
        input_dim = train_dataset.features.shape[1]
        output_dim = len(set(train_dataset.labels))
        
        # Create model
        model = create_model(
            input_dim, 
            output_dim, 
            config["model"]["hidden_layers"],
            device
        )
        
        # Load pre-trained parameters if available
        import_model_parameters(model, config["model"]["parameters_file"])
        
        # Create and return client
        return FlowerCSVClient(
            model, 
            train_loader, 
            test_loader,
            config["training"]["epochs"]
        )
    
    return ClientApp(client_fn=client_fn)