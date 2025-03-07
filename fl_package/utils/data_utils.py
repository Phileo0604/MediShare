import os
import numpy as np
from glob import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Add this after your existing imports
import torch
from torch.utils.data import Dataset

# Add this class definition
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


def load_datasets(data_path, target_column, batch_size=32, dataset_type="breast_cancer"):
    """
    Load datasets based on dataset type.
    
    Args:
        data_path: Path to the dataset CSV file or directory of Excel files
        target_column: Column name to use as target
        batch_size: Batch size for DataLoader
        dataset_type: Type of dataset to load
    
    Returns:
        train_dataset, test_dataset, train_loader, test_loader
    """
    if dataset_type.lower() == "parkinsons":
        # Use specialized loading for Parkinson's Excel files
        return load_parkinsons_dataset(data_path, target_column, batch_size)
    else:
        # Use the original loading function for other datasets
        # This could be implemented to handle the breast cancer dataset and other datasets
        return load_default_dataset(data_path, target_column, batch_size)

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