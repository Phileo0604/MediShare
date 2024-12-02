import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(config, input_dim=None, output_dim=None):
    if config["model_type"] == "feedforward":
        if input_dim is None or output_dim is None:
            raise ValueError("Input and output dimensions must be specified for feedforward models.")
        return FeedForwardNet(input_dim, output_dim)
    elif config["model_type"] == "cnn":
        if "num_classes" not in config:
            raise ValueError("Number of classes must be specified for CNN models.")
        return CNN(config["num_classes"])
    else:
        raise ValueError("Unsupported model type.")
