import torch.nn as nn

def create_recurrent(input_size, hidden_size, num_layers, num_classes):
    return nn.Sequential(
        nn.LSTM(input_size, hidden_size, num_layers, batch_first=True),
        nn.Linear(hidden_size, num_classes)
    )
