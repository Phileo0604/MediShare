# train_model.py
import argparse
import os
import sys
import json
import uuid
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_default_config(dataset_type):
    """Create a minimal config for the training script."""
    config = {
        "dataset": {
            "path": "",  # Will be filled by args.dataset_path
            "target_column": "target"
        },
        "training": {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "model": {
            "hidden_layers": [64, 32],
            "parameters_file": f"global_models/{dataset_type}_model.json"
        }
    }
    
    # Set appropriate defaults based on dataset type
    if dataset_type == "breast_cancer":
        config["dataset"]["target_column"] = "diagnosis"
    elif dataset_type == "parkinsons":
        config["dataset"]["target_column"] = "UPDRS"
    elif dataset_type == "reinopath":
        config["dataset"]["target_column"] = "class"
        
    return config

def import_models_based_on_dataset(dataset_type):
    """Import appropriate model classes based on dataset type."""
    if dataset_type == "breast_cancer":
        # Neural network model
        try:
            from models.nn_models import create_model, export_model_parameters
            return create_model, export_model_parameters
        except ImportError:
            print(f"Error: Cannot import neural network models.")
            sys.exit(1)
    elif dataset_type == "parkinsons":
        # XGBoost model
        try:
            from models.xgb_models import create_xgboost_model as create_model
            from models.nn_models import export_model_parameters
            return create_model, export_model_parameters
        except ImportError:
            print(f"Error: Cannot import XGBoost models.")
            sys.exit(1)
    elif dataset_type == "reinopath":
        # Specialized Reinopath model
        try:
            from models.reinopath_model import create_reinopath_model as create_model
            from models.nn_models import export_model_parameters
            return create_model, export_model_parameters
        except ImportError:
            print(f"Error: Cannot import Reinopath models.")
            sys.exit(1)
    else:
        # Default to neural network models
        try:
            from models.nn_models import create_model, export_model_parameters
            return create_model, export_model_parameters
        except ImportError:
            print(f"Error: Cannot import neural network models.")
            sys.exit(1)

def load_dataset(dataset_path, target_column, dataset_type):
    """
    Load and preprocess dataset based on its type and path
    """
    print(f"Loading dataset from {dataset_path}")
    
    # Load dataset based on file extension
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(dataset_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel files.")
    
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Using target column: {target_column}")
    
    # Clean up the dataset
    # Remove unnamed columns that might be added by pandas
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Handle missing values
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Check if target column exists in the dataset
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical target for classification tasks
    if dataset_type in ["breast_cancer", "reinopath"]:
        # Convert string labels to numeric
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
    
    # Feature preprocessing
    numeric_cols = X.select_dtypes(include=['number']).columns
    X = X[numeric_cols]  # Keep only numeric columns
    print(f"Using {len(numeric_cols)} numeric features")
    
    # Fill any remaining NaN values
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, X.shape[1]

def train_model(args):
    """
    Train a model based on dataset type and save it
    """
    dataset_type = args.dataset_type
    dataset_path = args.dataset_path
    config_path = args.config
    
    # Create or load configuration
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            config = create_default_config(dataset_type)
    else:
        print(f"Configuration file {config_path} not found, creating default configuration")
        config = create_default_config(dataset_type)
    
    # Update config with command line parameters if provided
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    
    # Update dataset path
    config["dataset"]["path"] = dataset_path
    
    # Get parameters from config
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    target_column = config["dataset"]["target_column"]
    
    print(f"Training parameters: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
    
    # Load dataset
    try:
        X_train, X_test, y_train, y_test, input_dim = load_dataset(dataset_path, target_column, dataset_type)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("PROGRESS: 0")
        return 1
    
    # Determine output dimension
    if dataset_type == "parkinsons":
        # Regression task
        output_dim = 1
        task = "regression"
    else:
        # Classification task
        output_dim = len(np.unique(y_train))
        if output_dim == 2:
            output_dim = 1  # Binary classification
        task = "classification"
    
    # Import model creation function based on dataset type
    try:
        create_model, export_model_parameters = import_models_based_on_dataset(dataset_type)
    except Exception as e:
        print(f"Error importing model classes: {e}")
        print("PROGRESS: 0")
        return 1
    
    # Create model
    print(f"Creating model for {dataset_type} with input_dim={input_dim}, output_dim={output_dim}")
    try:
        model = create_model(
            input_dim=input_dim, 
            output_dim=output_dim,
            hidden_layers=config["model"]["hidden_layers"],
            task=task
        )
    except Exception as e:
        print(f"Error creating model: {e}")
        print("PROGRESS: 0")
        return 1
    
    # Create PyTorch datasets and data loaders
    from torch.utils.data import TensorDataset, DataLoader
    import torch
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train) if task == "regression" else torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test) if task == "regression" else torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Train the model
    print(f"Starting model training with {epochs} epochs")
    
    # Use appropriate training function based on model type
    try:
        if hasattr(model, 'train') and callable(getattr(model, 'train')):
            # XGBoost or built-in training function
            for epoch in range(epochs):
                progress = int((epoch / epochs) * 100)
                print(f"PROGRESS: {progress}")
                model.train(train_loader, epochs=1, verbose=True)
        else:
            # PyTorch model
            import torch.optim as optim
            import torch.nn as nn
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
            
            for epoch in range(epochs):
                progress = int((epoch / epochs) * 100)
                print(f"PROGRESS: {progress}")
                
                # Training
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
    except Exception as e:
        print(f"Error during training: {e}")
        print("PROGRESS: 0")
        return 1
    
    # Evaluate model
    print("Evaluating model...")
    try:
        if hasattr(model, 'evaluate') and callable(getattr(model, 'evaluate')):
            # Model has built-in evaluation
            loss, metric = model.evaluate(test_loader)
            print(f"Test loss: {loss}, Metric: {metric}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    # Generate a unique filename
    unique_id = str(uuid.uuid4())
    if dataset_type == "breast_cancer":
        output_filename = f"breast_cancer_{unique_id}.json"
    elif dataset_type == "parkinsons":
        output_filename = f"parkinsons_{unique_id}.pkl"
    elif dataset_type == "reinopath":
        output_filename = f"reinopath_{unique_id}.pkl"
    else:
        output_filename = f"{dataset_type}_{unique_id}.json"
    
    # Create model directory if it doesn't exist
    model_dir = "global_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, output_filename)
    
    # Save model parameters
    try:
        export_model_parameters(model, model_path)
        print(f"Model parameters saved to {model_path}")
    except Exception as e:
        print(f"Error saving model parameters: {e}")
        print("PROGRESS: 0")
        return 1
    
    # Print model ID for the Java service to parse
    model_id = int(time.time())  # Using timestamp as a simple ID
    print(f"MODEL_ID: {model_id}")
    
    # Set progress to 100% to indicate completion
    print("PROGRESS: 100")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model based on dataset type")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--dataset-type", type=str, required=True, help="Type of dataset to use")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--job-id", type=str, default=None, help="Training job ID")
    
    args = parser.parse_args()
    sys.exit(train_model(args))