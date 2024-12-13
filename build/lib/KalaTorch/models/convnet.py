import torch.nn as nn

def create_convnet(input_channels, num_classes):
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),  # Adjust for your input size
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
