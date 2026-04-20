from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 42
LATENT_DIM = 100
IMG_ROWS = 28
IMG_COLS = 28
IMG_CHANNELS = 1
IMG_FLAT = IMG_ROWS * IMG_COLS * IMG_CHANNELS
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 400


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Generator(nn.Module):
    """Fully connected MNIST generator: z -> 784 pixels."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.Linear(1024, IMG_FLAT),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    """Fully connected MNIST discriminator: 784 pixels -> real/fake logit."""

    def __init__(self, img_flat: int = IMG_FLAT):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_flat, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class GANComponents:
    generator: Generator
    discriminator: Discriminator
    g_optimizer: torch.optim.Optimizer
    d_optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_gan(device: str | torch.device | None = None, latent_dim: int = LATENT_DIM, lr: float = 0.0002) -> GANComponents:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    return GANComponents(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        criterion=criterion,
        device=device,
    )


def make_dataloader(batch_size: int = DEFAULT_BATCH_SIZE, root: str = "./data") -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


@torch.no_grad()
def generate_images(generator: Generator, device: torch.device, n: int = 16, latent_dim: int = LATENT_DIM) -> torch.Tensor:
    generator.eval()
    z = torch.randn(n, latent_dim, device=device)
    imgs = generator(z).view(-1, 1, IMG_ROWS, IMG_COLS)
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1).cpu()


@torch.no_grad()
def generate_from_fixed_noise(generator: Generator, fixed_noise: torch.Tensor) -> torch.Tensor:
    generator.eval()
    imgs = generator(fixed_noise).view(-1, 1, IMG_ROWS, IMG_COLS)
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1).cpu()


def save_image_grid(images: torch.Tensor, path: str | Path, n_rows: int = 4, n_cols: int = 4, title: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 6))
    axes = axes.flatten()
    for ax, img in zip(axes, images[: n_rows * n_cols]):
        ax.imshow(img.squeeze(0), cmap="gray")
        ax.axis("off")
    for ax in axes[len(images[: n_rows * n_cols]) :]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_checkpoint(
    path: str | Path,
    generator: Generator,
    discriminator: Discriminator,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, List[float]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "history": history,
        },
        path,
    )
    return path


def load_generator_from_checkpoint(path: str | Path, device: str | torch.device | None = None) -> Generator:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    checkpoint = torch.load(path, map_location=device)
    generator = Generator().to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator


def train_gan(
    components: GANComponents,
    dataloader: DataLoader,
    epochs: int = DEFAULT_EPOCHS,
    latent_dim: int = LATENT_DIM,
    sample_dir: str | Path = "outputs/samples",
    checkpoint_dir: str | Path = "outputs/checkpoints",
    milestone_epochs: Tuple[int, ...] = (1, 30, 100, 400),
    progress_callback=None,
) -> Dict[str, List[float] | Dict[int, str]]:
    generator = components.generator
    discriminator = components.discriminator
    g_optimizer = components.g_optimizer
    d_optimizer = components.d_optimizer
    criterion = components.criterion
    device = components.device

    sample_dir = Path(sample_dir)
    checkpoint_dir = Path(checkpoint_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, List[float] | Dict[int, str]] = {
        "d_loss": [],
        "d_acc": [],
        "g_loss": [],
        "milestone_images": {},
    }

    fixed_noise = torch.randn(16, latent_dim, device=device)

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()

        epoch_d_loss: List[float] = []
        epoch_d_acc: List[float] = []
        epoch_g_loss: List[float] = []

        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(device).view(real_imgs.size(0), -1)
            batch_size = real_imgs.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train discriminator
            d_optimizer.zero_grad(set_to_none=True)
            real_logits = discriminator(real_imgs)
            d_loss_real = criterion(real_logits, real_labels)

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            fake_logits = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_logits, fake_labels)

            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            d_optimizer.step()

            with torch.no_grad():
                real_pred = (torch.sigmoid(real_logits) >= 0.5).float()
                fake_pred = (torch.sigmoid(fake_logits) < 0.5).float()
                d_acc = 0.5 * (real_pred.eq(real_labels).float().mean() + fake_pred.eq(torch.ones_like(fake_labels)).float().mean())

            # Train generator
            for p in discriminator.parameters():
                p.requires_grad = False

            g_optimizer.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, latent_dim, device=device)
            generated_imgs = generator(z)
            fool_logits = discriminator(generated_imgs)
            g_loss = criterion(fool_logits, real_labels)
            g_loss.backward()
            g_optimizer.step()

            for p in discriminator.parameters():
                p.requires_grad = True

            epoch_d_loss.append(float(d_loss.item()))
            epoch_d_acc.append(float(d_acc.item()))
            epoch_g_loss.append(float(g_loss.item()))

        history["d_loss"].append(sum(epoch_d_loss) / len(epoch_d_loss))
        history["d_acc"].append(sum(epoch_d_acc) / len(epoch_d_acc))
        history["g_loss"].append(sum(epoch_g_loss) / len(epoch_g_loss))

        if epoch in milestone_epochs or epoch == epochs:
            milestone = generate_from_fixed_noise(generator, fixed_noise)
            image_path = save_image_grid(milestone, sample_dir / f"epoch_{epoch:04d}.png", title=f"Epoch {epoch}")
            history["milestone_images"][epoch] = str(image_path)
            save_checkpoint(
                checkpoint_dir / "gan_checkpoint.pt",
                generator,
                discriminator,
                g_optimizer,
                d_optimizer,
                epoch,
                history,
            )

        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": epoch,
                    "d_loss": history["d_loss"][-1],
                    "d_acc": history["d_acc"][-1],
                    "g_loss": history["g_loss"][-1],
                }
            )

    return history
