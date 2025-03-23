import torch
import torch.nn as nn
import torch.optim as optim

# Import the model classes for type checking
from models.reinopath_model import ReinopathModel
# Import specialized trainers
from training.reinopath_trainer import train_reinopath, test_reinopath

def train(model, train_loader, epochs=10, learning_rate=0.001):
    """Train a model on the given data loader."""
    # Check if this is a Reinopath model
    if hasattr(model, 'train') and isinstance(model, ReinopathModel):
        return train_reinopath(model, train_loader, epochs=epochs)
    
    # Check if this is an XGBoost model
    elif hasattr(model, 'train') and hasattr(model, 'model') and hasattr(model.model, 'predict'):
        try:
            from training.xgb_trainer import train_xgboost
            return train_xgboost(model, train_loader, epochs=epochs)
        except ImportError:
            print("Warning: XGBoost trainer not found, falling back to default training.")
    
    # Default neural network training
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            for features, labels in train_loader:
                # Move data to the same device as model
                features = features.to(next(model.parameters()).device)
                labels = labels.to(next(model.parameters()).device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(features)
                loss = criterion(output, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


def test(model, test_loader):
    """Evaluate a model on the given test data loader."""
    # Check if this is a Reinopath model
    if hasattr(model, 'evaluate') and isinstance(model, ReinopathModel):
        return test_reinopath(model, test_loader)
    
    # Check if this is an XGBoost model
    elif hasattr(model, 'evaluate') and hasattr(model, 'model') and hasattr(model.model, 'predict'):
        try:
            from training.xgb_trainer import test_xgboost
            return test_xgboost(model, test_loader)
        except ImportError:
            print("Warning: XGBoost trainer not found, falling back to default evaluation.")
    
    # Default neural network evaluation
    else:
        criterion = nn.CrossEntropyLoss()
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in test_loader:
                # Move data to the same device as model
                features = features.to(next(model.parameters()).device)
                labels = labels.to(next(model.parameters()).device)
                
                # Forward pass
                output = model(features)
                loss = criterion(output, labels)
                
                # Compute metrics
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy