from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from gan_model import build_gan, count_parameters, make_dataloader, save_image_grid, train_gan, generate_images, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch GAN on MNIST.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed()
    out_dir = Path(args.out_dir)
    sample_dir = out_dir / "samples"
    checkpoint_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataloader = make_dataloader(batch_size=args.batch_size, root=args.data_dir)
    components = build_gan()

    g_total, g_trainable = count_parameters(components.generator)
    d_total, d_trainable = count_parameters(components.discriminator)

    print(f"Device: {components.device}")
    print(f"Generator params: {g_total:,} total / {g_trainable:,} trainable")
    print(f"Discriminator params: {d_total:,} total / {d_trainable:,} trainable")

    history = train_gan(
        components,
        dataloader,
        epochs=args.epochs,
        sample_dir=sample_dir,
        checkpoint_dir=checkpoint_dir,
    )

    history_df = pd.DataFrame(
        {
            "epoch": list(range(1, len(history["d_loss"]) + 1)),
            "d_loss": history["d_loss"],
            "d_acc": history["d_acc"],
            "g_loss": history["g_loss"],
        }
    )
    history_path = out_dir / "history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved history to {history_path}")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["d_loss"], label="Discriminator loss")
    plt.plot(history_df["epoch"], history_df["g_loss"], label="Generator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_plot = plots_dir / "loss_curves.png"
    fig.savefig(loss_plot, dpi=150)
    plt.close(fig)

    with torch.no_grad():
        final_images = generate_images(components.generator, components.device, n=16)
        save_image_grid(final_images, sample_dir / "final_grid.png", title="Final generated digits")

    print(f"Saved loss plot to {loss_plot}")
    print(f"Latest checkpoint: {checkpoint_dir / 'gan_checkpoint.pt'}")


if __name__ == "__main__":
    main()
