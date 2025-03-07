import torch
import torch.nn as nn
import torch.optim as optim

from training.xgb_trainer import train_xgboost, test_xgboost


def train(model, train_loader, epochs=10, learning_rate=0.001):
    """
    Generic training function that works with both PyTorch and XGBoost models.
    
    Args:
        model: The model to train (PyTorch nn.Module or XGBoostModel)
        train_loader: DataLoader with training data
        epochs: Number of training epochs
        learning_rate: Learning rate (for PyTorch models)
    """
    # Check if this is an XGBoost model
    if hasattr(model, 'model') and hasattr(model, 'train') and hasattr(model, 'get_parameters'):
        # XGBoost model
        train_xgboost(model, train_loader, epochs=epochs)
    else:
        # PyTorch model
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for features, labels in train_loader:
                # Move data to the same device as model
                device = next(model.parameters()).device
                features = features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(features)
                loss = criterion(output, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            # Calculate average loss per batch
            avg_loss = total_loss / max(batch_count, 1)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")


def test(model, test_loader):
    """
    Generic evaluation function that works with both PyTorch and XGBoost models.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader with test data
    
    Returns:
        loss, accuracy (or equivalent metrics)
    """
    # Check if this is an XGBoost model
    if hasattr(model, 'model') and hasattr(model, 'evaluate') and hasattr(model, 'get_parameters'):
        # XGBoost model
        return test_xgboost(model, test_loader)
    else:
        # PyTorch model
        criterion = nn.CrossEntropyLoss()
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in test_loader:
                # Move data to the same device as model
                device = next(model.parameters()).device
                features = features.to(device)
                labels = labels.to(device)
                
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