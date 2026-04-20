import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


# =========================================================
# DEVICE
# =========================================================

def get_device():
    """
    Returns the best available device:
    - CUDA for NVIDIA and many ROCm PyTorch builds
    - MPS for Apple Silicon
    - CPU fallback
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


def get_device_name():
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name(0)
    if DEVICE.type == "mps":
        return "Apple Metal (MPS)"
    return "CPU"


# =========================================================
# HYPERPARAMETERS
# =========================================================

LATENT_DIM = 100
IMAGE_SIZE = 28 * 28
IMAGE_SHAPE = (1, 28, 28)

LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)


# =========================================================
# MODELS
# =========================================================

class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), *IMAGE_SHAPE)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(IMAGE_SIZE, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        return self.model(img)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_models(latent_dim: int = LATENT_DIM):
    generator = Generator(latent_dim=latent_dim).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    return generator, discriminator


# =========================================================
# DATA
# =========================================================

def get_dataloader(batch_size=128, data_dir="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda")
    )
    return dataloader


# =========================================================
# HELPERS
# =========================================================

def set_requires_grad(model, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = requires_grad


def make_noise(batch_size: int, latent_dim: int = LATENT_DIM):
    return torch.randn(batch_size, latent_dim, device=DEVICE)


def denormalize(imgs):
    # From [-1, 1] back to [0, 1]
    return (imgs + 1) / 2


# =========================================================
# SAVE / LOAD
# =========================================================

def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "g_optimizer_state_dict": g_optimizer.state_dict(),
        "d_optimizer_state_dict": d_optimizer.state_dict(),
        "device": str(DEVICE),
    }, path)


def load_checkpoint(path, latent_dim: int = LATENT_DIM):
    generator, discriminator = build_models(latent_dim=latent_dim)

    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

    checkpoint = torch.load(path, map_location=DEVICE)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)

    generator.eval()
    discriminator.eval()

    return generator, discriminator, g_optimizer, d_optimizer, epoch


def save_generated_samples(generator, epoch, output_dir="outputs/samples", n=16):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        noise = make_noise(n)
        fake_imgs = generator(noise).cpu()
        fake_imgs = denormalize(fake_imgs)

    file_path = output_dir / f"epoch_{epoch:03d}.png"
    save_image(fake_imgs, file_path, nrow=4)
    return str(file_path)


def generate_images(generator, n=16):
    generator.eval()
    with torch.no_grad():
        noise = make_noise(n)
        fake_imgs = generator(noise).cpu()
        fake_imgs = denormalize(fake_imgs)
    return fake_imgs


# =========================================================
# TRAINING
# =========================================================

def train_gan(
    epochs=30,
    batch_size=128,
    latent_dim=LATENT_DIM,
    sample_interval=5,
    checkpoint_dir="outputs/checkpoints",
    sample_dir="outputs/samples",
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    dataloader = get_dataloader(batch_size=batch_size)

    generator, discriminator = build_models(latent_dim=latent_dim)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

    history = []

    for epoch in range(1, epochs + 1):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0

        generator.train()
        discriminator.train()

        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(DEVICE)
            batch_size_current = real_imgs.size(0)

            real_labels = torch.ones(batch_size_current, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size_current, 1, device=DEVICE)

            # -----------------------------
            # Train Discriminator
            # -----------------------------
            d_optimizer.zero_grad()

            real_output = discriminator(real_imgs)
            d_real_loss = criterion(real_output, real_labels)

            noise = make_noise(batch_size_current, latent_dim)
            fake_imgs = generator(noise)
            fake_output = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # -----------------------------
            # Train Generator
            # -----------------------------
            g_optimizer.zero_grad()

            noise = make_noise(batch_size_current, latent_dim)
            generated_imgs = generator(noise)
            validity = discriminator(generated_imgs)

            g_loss = criterion(validity, real_labels)
            g_loss.backward()
            g_optimizer.step()

            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()

        avg_g_loss = g_epoch_loss / len(dataloader)
        avg_d_loss = d_epoch_loss / len(dataloader)

        row = {
            "epoch": epoch,
            "g_loss": avg_g_loss,
            "d_loss": avg_d_loss,
        }
        history.append(row)

        if epoch == 1 or epoch % sample_interval == 0 or epoch == epochs:
            save_generated_samples(generator, epoch, sample_dir)

        save_checkpoint(
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            epoch,
            checkpoint_dir / "gan_checkpoint.pth"
        )

        torch.save(generator.state_dict(), checkpoint_dir / "generator_only.pth")

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}"
        )

    return {
        "generator": generator,
        "discriminator": discriminator,
        "history": history,
        "device": str(DEVICE),
        "device_name": get_device_name(),
        "generator_params": count_trainable_params(generator),
        "discriminator_params": count_trainable_params(discriminator),
    }


# =========================================================
# PLOT
# =========================================================

def plot_generated_images(generator, n=16):
    imgs = generate_images(generator, n=n)
    grid = make_grid(imgs, nrow=4, padding=2)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":
    generator, discriminator = build_models()

    print(f"Using device: {DEVICE}")
    print(f"Hardware: {get_device_name()}")
    print(f"Generator trainable params: {count_trainable_params(generator):,}")
    print(f"Discriminator trainable params: {count_trainable_params(discriminator):,}")

    result = train_gan(epochs=3, batch_size=128)

    print("Training finished.")
    print("History sample:", result["history"][:2])