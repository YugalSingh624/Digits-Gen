# Filename: model.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, digit_dim=10):
        super().__init__()
        self.label_emb = nn.Linear(digit_dim, noise_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]
