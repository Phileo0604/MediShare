import xgboost as xgb
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, f1_score

class ReinopathModel:
    """
    XGBoost model specifically designed for Reinopath diabetic retinopathy detection.
    """
    
    def __init__(self, input_dim, output_dim, params=None):
        """
        Initialize the Reinopath XGBoost model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes (1 for binary classification)
            params: XGBoost parameters
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Set specialized parameters for reinopath dataset
        if params is None:
            self.params = {
                'objective': 'binary:logistic',
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'auc',
                'scale_pos_weight': 3,  # Helps with class imbalance
                'gamma': 0.2,           # Minimum loss reduction for split
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
        Evaluate the model on test data with specialized metrics for diabetic retinopathy.
        
        Args:
            test_loader: DataLoader with test data
        
        Returns:
            Dictionary of metrics: AUC, accuracy, F1 score
        """
        X_test, y_test = self._dataloader_to_numpy(test_loader)
        dtest = xgb.DMatrix(X_test)
        
        # Make predictions (probabilities)
        y_pred_proba = self.model.predict(dtest)
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_class)
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred_class)
        
        # For compatibility with existing code, return loss (log loss) and accuracy
        # But also provide additional metrics
        loss = -np.mean(y_test * np.log(y_pred_proba + 1e-10) + 
                       (1 - y_test) * np.log(1 - y_pred_proba + 1e-10))
        
        # Print detailed metrics
        print(f"Evaluation metrics:")
        print(f"  AUC: {auc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return loss, accuracy
    
    def predict(self, X):
        """Make predictions on new data."""
        if isinstance(X, np.ndarray):
            # Already numpy
            pass
        else:
            # Convert to numpy
            X = X.detach().cpu().numpy() if hasattr(X, 'detach') else np.array(X)
        
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
    
    def feature_importance(self):
        """Get feature importance from the model."""
        if self.model is None:
            return {}
            
        importance = self.model.get_score(importance_type='gain')
        return importance
    
    def get_parameters(self):
        """
        Get model parameters as a list of numpy arrays.
        For XGBoost, we serialize the entire model.
        """
        if self.model is None:
            return []
        
        try:
            # Serialize the model to buffer
            model_data = self.model.save_raw()[4:]  # Skip the first 4 bytes (XGBoost header)
            
            # Return as a single numpy array
            return [np.frombuffer(model_data, dtype=np.uint8)]
        except:
            # Fallback if save_raw is not available (version compatibility)
            # Save to a temporary file and read bytes
            temp_file = "temp_model.bin"
            self.model.save_model(temp_file)
            with open(temp_file, "rb") as f:
                model_data = f.read()
            os.remove(temp_file)
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
        
        # Create a new model with the same parameters
        self.model = xgb.Booster(params=self.params)
        
        # Always use the file-based approach (more reliable)
        temp_file = "temp_model.bin"
        try:
            with open(temp_file, "wb") as f:
                f.write(model_data)
            
            # Load the model from file
            self.model.load_model(temp_file)
            
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            # If failed to load, reset the model
            self.reset_model()


def create_reinopath_model(input_dim, output_dim, params=None):
    """
    Factory function to create and initialize a Reinopath model.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output classes (1 for binary classification)
        params: Model parameters
    
    Returns:
        Initialized ReinopathModel
    """
    return ReinopathModel(input_dim, output_dim, params=params)