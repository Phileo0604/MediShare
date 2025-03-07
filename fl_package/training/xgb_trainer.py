import numpy as np
import xgboost as xgb

def train_xgboost(model, train_loader, epochs=10, learning_rate=None):
    """
    Train an XGBoost model on the given data.
    
    Args:
        model: The XGBoostModel to train
        train_loader: DataLoader with training data
        epochs: Number of training epochs (boosting rounds)
        learning_rate: Learning rate (not used - kept for API consistency)
    """
    if hasattr(model, 'train') and callable(model.train):
        # Train the model
        model.train(train_loader, epochs=epochs, verbose=True)
    else:
        raise TypeError("Model does not support XGBoost training")


def test_xgboost(model, test_loader):
    """
    Evaluate an XGBoost model on test data.
    
    Args:
        model: The XGBoostModel to evaluate
        test_loader: DataLoader with test data
    
    Returns:
        loss, accuracy (or equivalent metrics)
    """
    if hasattr(model, 'evaluate') and callable(model.evaluate):
        return model.evaluate(test_loader)
    else:
        raise TypeError("Model does not support XGBoost evaluation")


def predict_xgboost(model, features):
    """
    Make predictions with an XGBoost model.
    
    Args:
        model: The XGBoostModel to use
        features: Input features
    
    Returns:
        Predictions
    """
    if hasattr(model, 'predict') and callable(model.predict):
        return model.predict(features)
    else:
        raise TypeError("Model does not support XGBoost prediction")