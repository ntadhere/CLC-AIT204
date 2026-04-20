import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from gan_model import (
    DEVICE,
    LATENT_DIM,
    Generator,
    count_trainable_params,
    generate_images,
    get_device_name,
    train_gan,
)

st.set_page_config(page_title="PyTorch GAN Trainer", layout="wide")


# =========================================================
# HELPERS
# =========================================================

def resolve_device_choice(choice: str) -> str:
    if choice == "CPU":
        return "cpu"
    if choice == "GPU Only":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return DEVICE.type


def load_generator_only(path: str, latent_dim: int):
    model = Generator(latent_dim=latent_dim)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def show_image_grid(images_tensor, title="Generated Images"):
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    axes = axes.flatten()

    imgs = images_tensor.detach().cpu().numpy()

    for i, ax in enumerate(axes):
        if i < len(imgs):
            ax.imshow(imgs[i].squeeze(), cmap="gray")
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    st.pyplot(fig)


# =========================================================
# HEADER
# =========================================================

st.title("MNIST GAN Trainer")
st.write("Train a PyTorch GAN on MNIST using GPU when available.")

with st.expander("System Information", expanded=True):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Auto Detected Device", DEVICE.type.upper())

    with col_b:
        st.metric("Hardware", get_device_name())

    with col_c:
        if torch.cuda.is_available():
            st.success("GPU detected")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            st.success("Apple GPU detected")
        else:
            st.warning("No GPU detected")

# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("Training Controls")

device_option = st.sidebar.selectbox(
    "Training Device",
    ["Auto Detect", "GPU Only", "CPU"],
    index=0
)

epochs = st.sidebar.selectbox(
    "Epochs",
    [5, 10, 20, 30, 50, 100],
    index=2
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [32, 64, 128, 256],
    index=2
)

latent_dim = st.sidebar.selectbox(
    "Latent Dimension",
    [50, 100, 128, 256],
    index=1
)

sample_interval = st.sidebar.selectbox(
    "Sample Save Interval",
    [1, 2, 5, 10],
    index=2
)

num_generate = st.sidebar.selectbox(
    "Images to Generate",
    [4, 8, 16],
    index=2
)

project_dir = st.sidebar.text_input(
    "Project Output Folder",
    value="outputs"
)

train_button = st.sidebar.button("Start Training")
generate_button = st.sidebar.button("Generate Images From Saved Model")

# =========================================================
# MODEL INFO
# =========================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Generator Summary")
    gen_preview = Generator(latent_dim=latent_dim)
    st.write(f"Trainable parameters: **{count_trainable_params(gen_preview):,}**")

with col2:
    st.subheader("Run Settings")
    st.write(f"Device option: **{device_option}**")
    st.write(f"Epochs: **{epochs}**")
    st.write(f"Batch size: **{batch_size}**")
    st.write(f"Latent dimension: **{latent_dim}**")
    st.write(f"Output folder: **{project_dir}**")

# =========================================================
# TRAINING
# =========================================================

if train_button:
    chosen_device = resolve_device_choice(device_option)

    if device_option == "GPU Only" and chosen_device == "cpu":
        st.error("GPU Only selected, but no supported GPU was found.")
    else:
        st.info(f"Training will run on: **{chosen_device.upper()}**")

        checkpoint_dir = os.path.join(project_dir, "checkpoints")
        sample_dir = os.path.join(project_dir, "samples")

        with st.spinner("Training GAN..."):
            result = train_gan(
                epochs=epochs,
                batch_size=batch_size,
                latent_dim=latent_dim,
                sample_interval=sample_interval,
                checkpoint_dir=checkpoint_dir,
                sample_dir=sample_dir,
            )

        st.success("Training complete.")

        history_df = pd.DataFrame(result["history"])

        st.subheader("Training History")
        st.dataframe(history_df, use_container_width=True)

        st.subheader("Loss Curves")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history_df["epoch"], history_df["g_loss"], label="Generator Loss")
        ax.plot(history_df["epoch"], history_df["d_loss"], label="Discriminator Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("Training Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Device Used", result["device"].upper())
        c2.metric("Generator Params", f"{result['generator_params']:,}")
        c3.metric("Discriminator Params", f"{result['discriminator_params']:,}")

        sample_path = os.path.join(sample_dir, f"epoch_{epochs:03d}.png")
        if os.path.exists(sample_path):
            st.subheader("Latest Saved Sample")
            st.image(sample_path, caption=f"Saved samples at epoch {epochs}")

# =========================================================
# GENERATE FROM SAVED MODEL
# =========================================================

if generate_button:
    generator_path = os.path.join(project_dir, "checkpoints", "generator_only.pth")

    if not os.path.exists(generator_path):
        st.error("No saved generator model found. Train the GAN first.")
    else:
        st.info("Loading saved generator model...")

        model = load_generator_only(generator_path, latent_dim=latent_dim)
        imgs = generate_images(model, n=num_generate)

        st.subheader("Generated Images")
        if num_generate == 4:
            fig, axes = plt.subplots(2, 2, figsize=(5, 5))
            axes = axes.flatten()
            imgs_np = imgs.detach().cpu().numpy()
            for i, ax in enumerate(axes):
                ax.imshow(imgs_np[i].squeeze(), cmap="gray")
                ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
        elif num_generate == 8:
            fig, axes = plt.subplots(2, 4, figsize=(8, 4))
            axes = axes.flatten()
            imgs_np = imgs.detach().cpu().numpy()
            for i, ax in enumerate(axes):
                ax.imshow(imgs_np[i].squeeze(), cmap="gray")
                ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            show_image_grid(imgs, title="Generated MNIST Digits")

# =========================================================
# SAVED FILES
# =========================================================

st.subheader("Saved Files")

sample_folder = Path(project_dir) / "samples"
checkpoint_folder = Path(project_dir) / "checkpoints"

col3, col4 = st.columns(2)

with col3:
    st.write("**Sample Images**")
    if sample_folder.exists():
        sample_files = sorted(sample_folder.glob("*.png"))
        if sample_files:
            for f in sample_files[-5:]:
                st.write(f.name)
        else:
            st.write("No sample images yet.")
    else:
        st.write("Sample folder not created yet.")

with col4:
    st.write("**Checkpoint Files**")
    if checkpoint_folder.exists():
        checkpoint_files = sorted(checkpoint_folder.glob("*"))
        if checkpoint_files:
            for f in checkpoint_files:
                st.write(f.name)
        else:
            st.write("No checkpoint files yet.")
    else:
        st.write("Checkpoint folder not created yet.")