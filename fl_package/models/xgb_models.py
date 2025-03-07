import xgboost as xgb
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, mean_squared_error
import torch

class XGBoostModel:
    """
    XGBoost model wrapper for federated learning.
    This handles both classification and regression tasks.
    """
    
    def __init__(self, input_dim, output_dim, params=None, task='classification'):
        """
        Initialize the XGBoost model.
        
        Args:
            input_dim: Number of input features (not used by XGBoost but kept for API consistency)
            output_dim: Number of output classes or 1 for regression
            params: XGBoost parameters
            task: 'classification' or 'regression'
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task
        
        # Set default parameters based on task
        if params is None:
            if task == 'classification':
                if output_dim > 2:
                    # Multi-class classification
                    self.params = {
                        'objective': 'multi:softprob',
                        'num_class': output_dim,
                        'learning_rate': 0.1,
                        'max_depth': 5,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'eval_metric': 'mlogloss',
                        'tree_method': 'hist'
                    }
                else:
                    # Binary classification
                    self.params = {
                        'objective': 'binary:logistic',
                        'learning_rate': 0.1,
                        'max_depth': 5,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'eval_metric': 'logloss',
                        'tree_method': 'hist'
                    }
            else:
                # Regression
                self.params = {
                    'objective': 'reg:squarederror',
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'eval_metric': 'rmse',
                    'tree_method': 'hist'
                }
        else:
            self.params = params
        
        # Initialize model
        self.model = None
        self.reset_model()
    
    def reset_model(self):
        """Reset the model to its initial state."""
        self.model = xgb.Booster(params=self.params)
        self.model.set_param('nthread', -1)  # Use all CPU cores
    
    def train(self, train_loader, epochs=10, verbose=True):
        """
        Train the model on the given data.
        
        Args:
            train_loader: PyTorch DataLoader with features and labels
            epochs: Number of training rounds
            verbose: Whether to print training progress
        
        Returns:
            Training history
        """
        # Convert DataLoader to DMatrix
        X_train, y_train = self._dataloader_to_numpy(train_loader)
        
        if self.task == 'classification' and self.output_dim > 2:
            # Multi-class classification
            dtrain = xgb.DMatrix(X_train, label=y_train)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Train model
        results = {}
        for i in range(epochs):
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=1,
                xgb_model=self.model if i > 0 else None,
                evals=[(dtrain, 'train')],
                evals_result=results,
                verbose_eval=verbose
            )
            
            if verbose:
                # Get the last evaluation metric
                metric = list(results['train'].keys())[0]
                print(f"Epoch {i + 1}/{epochs}, {metric}: {results['train'][metric][-1]:.4f}")
        
        return results
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader with test data
        
        Returns:
            loss, accuracy (for classification) or mse (for regression)
        """
        X_test, y_test = self._dataloader_to_numpy(test_loader)
        dtest = xgb.DMatrix(X_test)
        
        # Make predictions
        y_pred = self.model.predict(dtest)
        
        if self.task == 'classification':
            if self.output_dim > 2:
                # Multi-class: predictions are probabilities, convert to class indices
                y_pred_class = np.argmax(y_pred.reshape(len(y_test), self.output_dim), axis=1)
                accuracy = accuracy_score(y_test, y_pred_class)
            else:
                # Binary: threshold at 0.5
                y_pred_class = (y_pred > 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred_class)
            
            # Return negative log loss as "loss" (approximate)
            loss = -np.mean(np.log(np.clip(y_pred, 1e-10, 1-1e-10)))
            return loss, accuracy
        else:
            # Regression: return MSE
            mse = mean_squared_error(y_test, y_pred)
            return mse, mse
    
    def predict(self, X):
        """Make predictions on new data."""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def _dataloader_to_numpy(self, dataloader):
        """Convert PyTorch DataLoader to numpy arrays."""
        features = []
        labels = []
        
        for batch_features, batch_labels in dataloader:
            if hasattr(batch_features, 'numpy'):
                # PyTorch Tensor
                features.append(batch_features.detach().cpu().numpy())
                labels.append(batch_labels.detach().cpu().numpy())
            else:
                # Already numpy
                features.append(batch_features)
                labels.append(batch_labels)
        
        X = np.vstack(features)
        y = np.concatenate(labels)
        
        return X, y
    
    def get_parameters(self):
        """
        Get model parameters as a list of numpy arrays.
        For XGBoost, we serialize the entire model.
        """
        if self.model is None:
            return []
        
        # Serialize the model to buffer
        model_data = self.model.save_raw()[4:]  # Skip the first 4 bytes (XGBoost header)
        
        # Return as a single numpy array
        return [np.frombuffer(model_data, dtype=np.uint8)]
    
    def set_parameters(self, parameters):
        """
        Set model parameters from a list of numpy arrays.
        For XGBoost, we deserialize the entire model.
        """
        if not parameters or len(parameters) == 0:
            self.reset_model()
            return
        
        # Convert first parameter back to bytes
        model_data = parameters[0].tobytes()
        
        # Create a new model
        self.model = xgb.Booster(params=self.params)
        
        # Load the model from raw bytes
        try:
            self.model.load_model_from_buffer(model_data)
        except Exception as e:
            print(f"Error loading XGBoost model: {e}")
            self.reset_model()


def create_xgboost_model(input_dim, output_dim, params=None, task="classification"):
    """
    Factory function to create and initialize an XGBoost model.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output classes or 1 for regression
        params: XGBoost parameters
        task: 'classification' or 'regression'
    
    Returns:
        Initialized XGBoost model wrapper
    """
    return XGBoostModel(input_dim, output_dim, params=params, task=task)


def get_model_path(dataset_type):
    """
    Get the appropriate model path based on dataset type.
    This allows dataset-specific global models.
    """
    base_dir = "global_models"
    os.makedirs(base_dir, exist_ok=True)
    
    if dataset_type.lower() == "breast_cancer":
        return os.path.join(base_dir, "breast_cancer_model.json")
    elif dataset_type.lower() == "parkinsons":
        return os.path.join(base_dir, "parkinsons_model.pkl")
    elif dataset_type.lower() == "third_dataset":
        return os.path.join(base_dir, "third_dataset_model.pkl")
    else:
        # Default path
        return os.path.join(base_dir, "global_model.json")