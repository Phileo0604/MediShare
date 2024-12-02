import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from torchvision.datasets import ImageFolder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class CSVDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def preprocess_data(dataset_type, dataset_path, target_column=None):
    if dataset_type == "csv":
        # Load the CSV file
        data = pd.read_csv(dataset_path)
        print("Initial dataset shape:", data.shape)  # Debugging

        # Check for missing values in each column
        print("Missing values per column:")
        print(data.isnull().sum())  # This will show how many missing values there are in each column

        # Drop columns that are entirely NaN or unwanted columns (e.g., 'Unnamed: 32')
        data = data.dropna(axis=1, how='all')  # Drop columns with all NaN values
        data = data.drop(columns=["Unnamed: 32"], errors='ignore')  # Drop the unwanted column

        print("Dataset shape after cleaning:", data.shape)  # Debugging

        # Check if the target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
        # Encode the target variable (e.g., diagnosis)
        label_encoder = LabelEncoder()
        data[target_column] = label_encoder.fit_transform(data[target_column])
        print(f"Encoded target column: {data[target_column].unique()}")  # Debugging

        # Separate features and target
        X = data.drop(columns=[target_column, "id"])  # Drop 'id' column as it is not a feature
        y = data[target_column].values
        print("Feature shape:", X.shape, "Target shape:", y.shape)  # Debugging

        # Normalize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Scaled feature shape:", X_scaled.shape)  # Debugging

        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        print("Train feature shape:", X_train.shape, "Test feature shape:", X_test.shape)  # Debugging

        return X_train, X_test, y_train, y_test



def load_dataset(config):
    """Load and preprocess the dataset based on the provided config."""
    X_train, X_test, y_train, y_test = preprocess_data(config["dataset_type"], config["dataset_path"], config["target_column"])
    
    # Create dataset objects
    train_data = CSVDataset(X_train, y_train)
    test_data = CSVDataset(X_test, y_test)

    # Return DataLoaders for batching
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config["batch_size"])

    return train_loader, test_loader  # Ensure you return the loaders here

