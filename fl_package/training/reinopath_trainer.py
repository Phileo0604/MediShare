import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

def train_reinopath(model, train_loader, epochs=10, learning_rate=None, early_stopping=True, patience=3):
    """
    Train a Reinopath model on diabetic retinopathy data.
    
    Args:
        model: ReinopathModel to train
        train_loader: DataLoader with training data
        epochs: Number of training epochs (boosting rounds)
        learning_rate: Learning rate (not used - kept for API consistency)
        early_stopping: Whether to use early stopping
        patience: Number of epochs to wait before early stopping
    """
    start_time = time.time()
    
    if hasattr(model, 'train') and callable(model.train):
        # Train the model
        print(f"Starting training Reinopath model for {epochs} epochs...")
        history = model.train(train_loader, epochs=epochs, verbose=True)
        
        # Print feature importance
        if hasattr(model, 'feature_importance') and callable(model.feature_importance):
            try:
                importance = model.feature_importance()
                if importance:
                    print("\nFeature Importance (top 10):")
                    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    for feature, importance in sorted_importance:
                        print(f"  {feature}: {importance:.4f}")
            except:
                pass
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return history
    else:
        raise TypeError("Model does not support Reinopath training")


def test_reinopath(model, test_loader):
    """
    Evaluate a Reinopath model on test data with specialized metrics for diabetic retinopathy.
    
    Args:
        model: ReinopathModel to evaluate
        test_loader: DataLoader with test data
    
    Returns:
        loss, accuracy
    """
    if hasattr(model, 'evaluate') and callable(model.evaluate):
        return model.evaluate(test_loader)
    else:
        raise TypeError("Model does not support Reinopath evaluation")


def predict_reinopath(model, features):
    """
    Make predictions with a Reinopath model.
    
    Args:
        model: ReinopathModel to use
        features: Input features
    
    Returns:
        Predictions (probabilities of diabetic retinopathy)
    """
    if hasattr(model, 'predict') and callable(model.predict):
        return model.predict(features)
    else:
        raise TypeError("Model does not support Reinopath prediction")


def explain_reinopath_predictions(model, X, feature_names=None):
    """
    Generate explanations for Reinopath model predictions.
    
    Args:
        model: Trained ReinopathModel
        X: Features to explain (numpy array)
        feature_names: List of feature names
    
    Returns:
        Feature importance for this prediction
    """
    try:
        # Get feature importance from model
        importance = model.feature_importance()
        
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Map feature indices to names
        named_importance = {}
        for feature, imp in importance.items():
            # XGBoost feature names are f0, f1, etc.
            idx = int(feature.replace('f', ''))
            if idx < len(feature_names):
                named_importance[feature_names[idx]] = imp
        
        return named_importance
    except:
        print("Feature importance explanation not available")
        return {}