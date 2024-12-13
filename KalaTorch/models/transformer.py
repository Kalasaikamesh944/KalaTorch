import torch.nn as nn

def create_transformer(input_dim, num_heads, num_layers, num_classes):
    encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    return nn.Sequential(
        transformer,
        nn.Linear(input_dim, num_classes)
    )
