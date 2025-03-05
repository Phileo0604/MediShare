import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


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


def load_datasets(dataset_path, target_column, batch_size, test_size=0.2):
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