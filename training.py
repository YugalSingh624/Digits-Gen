# Filename: train_digit_generator.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Generator (Simple CNN Decoder)
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

# One-hot encode digits
def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels].to(labels.device)

# Training
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    generator = Generator().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=0.001)

    for epoch in range(10):
        for real_imgs, labels in dataloader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            noise = torch.randn(batch_size, 64).to(device)
            onehot = one_hot(labels).to(device)
            fake_imgs = generator(noise, onehot)

            real_imgs = real_imgs.view(-1, 784)
            loss = criterion(fake_imgs.view(-1, 784), real_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/10 | Loss: {loss.item():.4f}")

    torch.save(generator.state_dict(), "digit_generator.pth")

if __name__ == "__main__":
    train()
