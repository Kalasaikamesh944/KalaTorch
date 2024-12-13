# **KalaTorch**  
**KalaTorch** is a simplified AI framework built on PyTorch. It provides implementations for a variety of models across multiple AI domains, including Computer Vision, Natural Language Processing (NLP), Reinforcement Learning (RL), and Generative Models.

All models and utilities are contained in a single script: `kala_torch.py`.

---

## **Features**
- **Computer Vision**:
  - Image Classification (`ResNet`, `SimpleCNN`)
  - Object Detection (`YOLOTiny`)
  - Semantic Segmentation (`UNet`)
- **Natural Language Processing**:
  - Text Classification (`BERTClassifier`)
  - Text Generation (`GPTGenerator`)
- **Reinforcement Learning**:
  - Actor-Critic framework
- **Generative Models**:
  - GAN (`GANGenerator`)
  - Placeholder for Diffusion Models
- **Training Utilities**:
  - Generic trainer class to simplify model training.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/Kalasaikamesh944/KalaTorch.git
   cd kalatorch
   ```

2. Install required dependencies:
   ```bash
   pip install torch torchvision transformers pandas scikit-learn tqdm
   ```

---

## **Usage**

1. **Import KalaTorch**:
   ```python
   from kala_torch import *
   ```

2. **Examples**:

### **1. Vision: Image Classification**
```python
from kala_torch import get_resnet_model, Trainer, load_data

# Initialize a ResNet model for 10 classes
model = get_resnet_model(num_classes=10)

# Load data (CSV should contain features and a 'label' column)
train_loader, val_loader = load_data("data.csv", label_column="label")

# Train the model
trainer = Trainer(model, optimizer='adam', loss_fn='cross_entropy')
trainer.train(train_loader, val_loader, epochs=5)
```

---

### **2. NLP: Text Generation**
```python
from kala_torch import GPTGenerator

# Initialize the GPT model
gpt = GPTGenerator()

# Generate text
prompt = "Once upon a time"
output = gpt.generate(prompt)
print("Generated Text:", output)
```

---

### **3. Reinforcement Learning**
```python
from kala_torch import ActorCritic

# Initialize the Actor-Critic model
state_dim = 4
action_dim = 2
rl_model = ActorCritic(state_dim=state_dim, action_dim=action_dim)

# Example state
state = torch.randn(state_dim)
action = rl_model.select_action(state)
print("Selected Action:", action)
```

---

### **4. Generative Models: GAN**
```python
from kala_torch import GANGenerator

# Initialize GAN generator
latent_dim = 100
output_dim = 28 * 28  # For MNIST-sized images
gan = GANGenerator(latent_dim=latent_dim, output_dim=output_dim)

# Generate a sample
z = torch.randn(1, latent_dim)
generated_sample = gan(z)
print("Generated Sample Shape:", generated_sample.shape)
```

---

## **Modules**

### **1. Computer Vision**
| **Model**           | **Description**                               |
|----------------------|-----------------------------------------------|
| `SimpleCNN`          | A simple CNN for image classification.       |
| `get_resnet_model`   | Pretrained ResNet with customizable classes. |
| `YOLOTiny`           | A lightweight object detection model.        |
| `UNet`               | A semantic segmentation model.               |

### **2. Natural Language Processing**
| **Model**            | **Description**                              |
|----------------------|-----------------------------------------------|
| `BERTClassifier`     | Transformer-based text classification.       |
| `GPTGenerator`       | HuggingFace GPT model for text generation.   |

### **3. Reinforcement Learning**
| **Model**            | **Description**                              |
|----------------------|-----------------------------------------------|
| `ActorCritic`        | Policy gradient RL model with Actor-Critic.  |

### **4. Generative Models**
| **Model**            | **Description**                              |
|----------------------|-----------------------------------------------|
| `GANGenerator`       | Generator for GAN-based image generation.    |
| `DiffusionModel`     | Placeholder for diffusion-based generation.  |

---

## **Trainer Utility**

The `Trainer` class simplifies model training. It supports:
- Custom loss functions (`cross_entropy`, `mse`)
- Optimizers (`adam`, `sgd`)
- Automatic validation after each epoch.

### **Example: Using the Trainer**
```python
from kala_torch import Trainer, SimpleCNN, load_data

# Initialize model and data loaders
model = SimpleCNN(num_classes=10)
train_loader, val_loader = load_data("data.csv", label_column="label")

# Train the model
trainer = Trainer(model, optimizer='adam', loss_fn='cross_entropy')
trainer.train(train_loader, val_loader, epochs=5)
```

---

## **File Structure**
All functionality resides in `kala_torch.py`.

---

## **License**
This project is licensed under the MIT License.

---

## **Contributing**
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

Feel free to open issues for bug reports or feature requests.

---

## **Support**
For questions or support, reach out to **[saikamesh.y@gmail.com]**.

