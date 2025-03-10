import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ReinopathDataset(Dataset):
    """Dataset class for Reinopath diabetic retinopathy data."""
    
    def __init__(self, features, labels):
        """
        Initialize dataset.
        
        Args:
            features: Numpy array of features
            labels: Numpy array of labels
        """
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_reinopath_dataset(data_path, target_column="class", batch_size=32, test_size=0.2):
    """
    Load and preprocess the Reinopath diabetic retinopathy dataset.
    
    Args:
        data_path: Path to the dataset CSV file
        target_column: Column to use as target (default: "class")
        batch_size: Batch size for DataLoader
        test_size: Proportion of data to use for testing
    
    Returns:
        train_dataset, test_dataset, train_loader, test_loader
    """
    # Load the dataset
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded Reinopath dataset with {df.shape[0]} samples and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Extract features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Define feature names
    feature_names = X.columns.tolist()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Create datasets
    train_dataset = ReinopathDataset(X_train_tensor, y_train_tensor)
    test_dataset = ReinopathDataset(X_test_tensor, y_test_tensor)
    
    # Add feature information for later use
    train_dataset.feature_names = feature_names
    test_dataset.feature_names = feature_names
    
    # Add original data shapes
    train_dataset.features_shape = X_train.shape
    test_dataset.features_shape = X_test.shape
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Split into {len(train_dataset)} training and {len(test_dataset)} testing samples")
    print(f"Features shape: {X_train.shape[1]}")
    print(f"Class distribution - Training: {np.bincount(y_train.astype(int))}, Testing: {np.bincount(y_test.astype(int))}")
    
    return train_dataset, test_dataset, train_loader, test_loader