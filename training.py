import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator with label embedding
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

# Discriminator with label embedding
class Discriminator(nn.Module):
    def __init__(self, digit_dim=10):
        super().__init__()
        self.label_emb = nn.Linear(digit_dim, 784)
        self.model = nn.Sequential(
            nn.Linear(784 * 2, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([img.view(img.size(0), -1), label_embedding], dim=1)
        return self.model(x)

def one_hot(labels, num_classes=10):
    return torch.eye(num_classes, device=labels.device)[labels]

def save_samples(generator, epoch, sample_dir="samples", fixed_noise=None, fixed_labels=None):
    generator.eval()
    os.makedirs(sample_dir, exist_ok=True)
    if fixed_noise is None or fixed_labels is None:
        z = torch.randn(100, 100, device=device)
        labels = torch.tensor([i % 10 for i in range(100)], device=device)
    else:
        z = fixed_noise
        labels = fixed_labels

    onehot = one_hot(labels)
    samples = generator(z, onehot)
    save_image(samples, f"{sample_dir}/epoch_{epoch}.png", nrow=10, normalize=True)
    generator.train()

def train():
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss and optimizers
    criterion = nn.MSELoss()
    opt_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Fixed noise for tracking progress
    fixed_z = torch.randn(100, 100, device=device)
    fixed_labels = torch.tensor([i % 10 for i in range(100)], device=device)
    fixed_onehot = one_hot(fixed_labels)

    for epoch in range(1, 51):  # Train for 50 epochs
        for real_imgs, labels in dataloader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            onehot_labels = one_hot(labels)

            # Label smoothing
            real_targets = torch.ones(batch_size, 1, device=device) * 0.9
            fake_targets = torch.zeros(batch_size, 1, device=device) + 0.1

            # Train Discriminator
            z = torch.randn(batch_size, 100, device=device)
            fake_imgs = generator(z, onehot_labels).detach()

            d_real = discriminator(real_imgs, onehot_labels)
            d_fake = discriminator(fake_imgs, onehot_labels)
            d_loss = criterion(d_real, real_targets) + criterion(d_fake, fake_targets)

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # Train Generator
            z = torch.randn(batch_size, 100, device=device)
            fake_imgs = generator(z, onehot_labels)
            g_loss = criterion(discriminator(fake_imgs, onehot_labels), real_targets)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        print(f"Epoch {epoch:>2} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        save_samples(generator, epoch, fixed_noise=fixed_z, fixed_labels=fixed_labels)

    torch.save(generator.state_dict(), "cgan_digit_generator.pth")

if __name__ == "__main__":
    train()
