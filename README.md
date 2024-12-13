# KalaTorch
# Author : ``` N V R K SAI KAMESH SHARMA ```
# Install using pip
``` pip install kalatorch ```
---

**KalaTorch** is a high-level PyTorch framework that simplifies the implementation of AI models and neural networks. This framework is designed to help developers quickly build, train, and evaluate machine learning models with minimal boilerplate code.

---

## Features

- **Predefined Architectures**: Easily create convolutional, recurrent, and transformer-based neural networks.
- **Dataset Management**: Simplifies dataset creation and loading.
- **Training Utilities**: High-level APIs for model training and evaluation.
- **Modular Design**: Extend and customize components for various tasks.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/KalaTorch.git
cd KalaTorch
pip install -r requirements.txt
```

Or, install directly via pip (coming soon):

```bash
pip install kalatorch
```

---

## Usage

### Example: Training a Convolutional Neural Network

```python
import torch
import torch.nn as nn
from KalaTorch.models import create_convnet
from KalaTorch.datasets import create_data_loader
from KalaTorch.training import KaloTrainer
import torch.optim as optim

# Define dataset
X = torch.randn(100, 1, 28, 28)  # 100 samples of 28x28 grayscale images
y = torch.randint(0, 10, (100,))  # Labels for 10 classes

# Create data loaders
train_loader = create_data_loader(X[:80], y[:80], batch_size=16)
val_loader = create_data_loader(X[80:], y[80:], batch_size=16)

# Create a convolutional model
model = create_convnet(input_channels=1, num_classes=10)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Initialize the trainer
trainer = KaloTrainer(model, optimizer, criterion)

# Train the model
trainer.train(train_loader, epochs=5)

# Evaluate the model
trainer.evaluate(val_loader)
```

---

## API Reference

### 1. **Models**

#### `create_convnet(input_channels, num_classes)`
- Builds a convolutional neural network.
- **Parameters:**
  - `input_channels` (int): Number of input channels.
  - `num_classes` (int): Number of output classes.

#### `create_recurrent(input_size, hidden_size, num_layers, num_classes)`
- Builds a recurrent neural network.
- **Parameters:**
  - `input_size` (int): Dimensionality of input features.
  - `hidden_size` (int): Number of hidden units per layer.
  - `num_layers` (int): Number of RNN layers.
  - `num_classes` (int): Number of output classes.

#### `create_transformer(input_dim, num_heads, num_layers, num_classes)`
- Builds a transformer-based model.
- **Parameters:**
  - `input_dim` (int): Dimensionality of input features.
  - `num_heads` (int): Number of attention heads.
  - `num_layers` (int): Number of encoder layers.
  - `num_classes` (int): Number of output classes.

### 2. **Datasets**

#### `create_data_loader(data, labels, batch_size=32, shuffle=True)`
- Creates a PyTorch DataLoader for the given dataset.
- **Parameters:**
  - `data` (torch.Tensor): Input features.
  - `labels` (torch.Tensor): Ground truth labels.
  - `batch_size` (int): Size of each batch (default: 32).
  - `shuffle` (bool): Whether to shuffle the data (default: True).

### 3. **Training**

#### `KaloTrainer`
- A class for managing training and evaluation workflows.

**Methods:**

- `train(train_loader, epochs=10)`
  - Trains the model for the specified number of epochs.
  - **Parameters:**
    - `train_loader`: DataLoader for training data.
    - `epochs` (int): Number of training epochs (default: 10).

- `evaluate(val_loader)`
  - Evaluates the model on the validation dataset.
  - **Parameters:**
    - `val_loader`: DataLoader for validation data.

---

## Project Structure

```
KalaTorch/
├── datasets/                 # Dataset utilities
│   ├── __init__.py
│   ├── dataloader.py
├── models/                   # Predefined models
│   ├── __init__.py
│   ├── convnet.py
│   ├── recurrent.py
│   ├── transformer.py
├── training/                 # Training utilities
│   ├── __init__.py
│   ├── trainer.py
├── utils/                    # Optional helper functions
│   ├── __init__.py
├── tests/                    # Unit tests
│   ├── test_trainer.py
└── train.py                  # Example training script
```

---

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Links

- [Documentation](https://github.com/Kalasaikamesh944/KalaTorch#readme)
- [Issue Tracker](https://github.com/Kalasaikamesh944/KalaTorch/issues)
- [Source Code](https://github.com/Kalasaikamesh944/KalaTorch)
- 

