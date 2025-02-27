import torch
import torch.nn as nn
import torch.optim as optim


def train(model, train_loader, epochs, learning_rate=0.001):
    """Train a model on the given data loader."""
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