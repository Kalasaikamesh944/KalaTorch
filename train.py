import unittest
import torch
from KalaTorch.models import create_convnet
from KalaTorch.training import KaloTrainer
from KalaTorch.datasets import create_data_loader
import torch.nn as nn
import torch.optim as optim

class TestKaloTrainer(unittest.TestCase):
    def test_training(self):
        model = create_convnet(1, 10)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        trainer = KaloTrainer(model, optimizer, criterion)

        X = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 10, (100,))
        train_loader = create_data_loader(X[:80], y[:80])
        val_loader = create_data_loader(X[80:], y[80:])

        trainer.train(train_loader, epochs=1)
        accuracy = trainer.evaluate(val_loader)
        self.assertGreaterEqual(accuracy, 0)

if __name__ == "__main__":
    unittest.main()
