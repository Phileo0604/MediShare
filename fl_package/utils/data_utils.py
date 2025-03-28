import os
import numpy as np
from glob import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class CustomDataset(Dataset):
    """Custom dataset class that holds features and labels separately."""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CSVDataset(Dataset):
    """Dataset class for loading and preprocessing CSV data."""
    
    def __init__(self, data: pd.DataFrame, target_column: str):
        # Remove unnamed columns that might be added by pandas
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        
        # Split features and labels
        self.features = data.drop(columns=['id', target_column], errors='ignore')
        self.labels = data[target_column]

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        self.features = pd.DataFrame(imputer.fit_transform(self.features), 
                                    columns=self.features.columns)

        # Encode target labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

        # Standardize features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


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
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


def load_default_dataset(dataset_path, target_column, batch_size, test_size=0.2):
    """Load and split dataset into training and testing sets."""
    try:
        data = pd.read_csv(dataset_path)
        
        # Split data into train and test sets
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=42
        )
        
        # Create dataset objects
        train_dataset = CSVDataset(train_data, target_column)
        test_dataset = CSVDataset(test_data, target_column)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size
        )
        
        return train_dataset, test_dataset, train_loader, test_loader
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def create_dummy_dataset(dataset_type, batch_size=32):
    """
    Create a dummy dataset when real data isn't available.
    Used only for model initialization, not for training.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create minimal datasets based on dataset type
    if dataset_type == "breast_cancer":
        features = torch.zeros((10, 30), dtype=torch.float32)
        labels = torch.zeros(10, dtype=torch.long)
    elif dataset_type == "parkinsons":
        features = torch.zeros((10, 16), dtype=torch.float32)
        labels = torch.zeros(10, dtype=torch.float32)
    elif dataset_type == "reinopath":
        features = torch.zeros((10, 19), dtype=torch.float32)
        labels = torch.zeros(10, dtype=torch.long)
    else:
        # Generic default
        features = torch.zeros((10, 10), dtype=torch.float32)
        labels = torch.zeros(10, dtype=torch.long)
    
    # Create dataset and loaders
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    print(f"Created dummy dataset for {dataset_type} with shape {features.shape}")
    return dataset, dataset, loader, loader

def load_parkinsons_dataset(data_path, target_column, batch_size=32, test_size=0.2):
    """
    Load and preprocess the Parkinson's disease dataset from multiple Excel files.
    
    Args:
        data_path: Directory containing the Excel files
        target_column: Column name to use as target (e.g., 'UPDRS')
        batch_size: Batch size for DataLoader
        test_size: Proportion of data to use for testing
    
    Returns:
        train_dataset, test_dataset, train_loader, test_loader
    """
    # Get all Excel files in the directory
    if os.path.isdir(data_path):
        excel_files = glob(os.path.join(data_path, "gait_parkinsons_*.xlsx"))
    else:
        # If data_path points to a specific file
        excel_files = [data_path]
    
    if not excel_files:
        raise ValueError(f"No Excel files found at {data_path}")
    
    # Load and concatenate all Excel files
    dataframes = []
    for file in excel_files:
        df = pd.read_excel(file)
        dataframes.append(df)
    
    # Combine all dataframes
    df = pd.concat(dataframes, ignore_index=True)
    
    # Fill missing values - use median for numerical columns
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        # Fill missing with most common value
        df[col] = df[col].fillna(df[col].mode()[0])
        
        # Encode categorical variables
        if col != target_column:  # Don't encode the target if it's categorical here
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
    
    # Extract features and target
    X = df.drop(columns=[target_column], errors='ignore')
    
    if target_column in df.columns:
        y = df[target_column]
        
        # If target is categorical, encode it
        if y.dtype == 'object' or y.dtype == 'category':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
    else:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Convert to numeric and handle remaining non-numeric columns
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=torch.float32)
    
    # Create datasets
    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, test_dataset, train_loader, test_loader


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
        if os.path.isdir(data_path):
            # If a directory is provided, look for CSV files
            for file in os.listdir(data_path):
                if file.lower().endswith('.csv') and 'reinopath' in file.lower():
                    data_path = os.path.join(data_path, file)
                    break
        
        df = pd.read_csv(data_path)
        print(f"Loaded Reinopath dataset with {df.shape[0]} samples and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Extract features and target
    X = df.drop(columns=[target_column], errors='ignore')
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
    y_train_tensor = torch.LongTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test.values)
    
    # Create datasets
    train_dataset = ReinopathDataset(X_train, y_train.values)
    test_dataset = ReinopathDataset(X_test, y_test.values)
    
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


def load_datasets(data_path, target_column, batch_size=32, dataset_type="breast_cancer"):
    """
    Load datasets based on dataset type.
    
    Args:
        data_path: Path to the dataset CSV file or directory
        target_column: Column name to use as target
        batch_size: Batch size for DataLoader
        dataset_type: Type of dataset to load
    
    Returns:
        train_dataset, test_dataset, train_loader, test_loader
    """
    try:
        if dataset_type.lower() == "parkinsons":
            # Use specialized loading for Parkinson's Excel files
            return load_parkinsons_dataset(data_path, target_column, batch_size)
        elif dataset_type.lower() == "reinopath":
            # Use specialized loading for Reinopath dataset
            return load_reinopath_dataset(data_path, target_column, batch_size)
        else:
            # Use the original loading function for other datasets
            # (e.g., breast cancer dataset)
            return load_default_dataset(data_path, target_column, batch_size)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        
        # If the data file doesn't exist but we're in parameter-only mode,
        # return a dummy dataset that won't be used for training
        if "No such file or directory" in str(e):
            print(f"Using dummy dataset for parameter-only mode")
            return create_dummy_dataset(dataset_type, batch_size)
        
        # For other errors, re-raise
        raise