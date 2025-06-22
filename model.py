# Filename: model.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, digit_embedding_dim=10, noise_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + digit_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, noise, digit_onehot):
        x = torch.cat([noise, digit_onehot], dim=1)
        img = self.fc(x)
        return img.view(-1, 1, 28, 28)

def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]
