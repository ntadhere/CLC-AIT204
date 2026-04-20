from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from gan_model import (
    build_gan,
    count_parameters,
    generate_images,
    load_generator_from_checkpoint,
    make_dataloader,
    train_gan,
    set_seed,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints" / "gan_checkpoint.pt"


st.set_page_config(page_title="MNIST GAN - PyTorch", layout="wide")
st.title("MNIST GAN Trainer - PyTorch + Streamlit")
st.caption("Converted from your TensorFlow notebook to a PyTorch GAN with explicit trainable parameters.")

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with st.sidebar:
    st.header("Training settings")
    epochs = st.slider("Epochs", min_value=1, max_value=400, value=30, step=1)
    batch_size = st.selectbox("Batch size", [32, 64, 128, 256], index=2)
    data_dir = st.text_input("Data directory", value=str(PROJECT_DIR / "data"))
    sample_interval_hint = st.caption("Milestones are saved at epoch 1, 30, 100, 400, and the final epoch.")

st.subheader("GPU status")
if torch.cuda.is_available():
    st.success(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    st.warning("No CUDA GPU detected. Training will use CPU.")

components = build_gan(device=device)
g_total, g_trainable = count_parameters(components.generator)
d_total, d_trainable = count_parameters(components.discriminator)

c1, c2 = st.columns(2)
with c1:
    st.metric("Generator trainable weights", f"{g_trainable:,}")
    st.metric("Generator total weights", f"{g_total:,}")
with c2:
    st.metric("Discriminator trainable weights", f"{d_trainable:,}")
    st.metric("Discriminator total weights", f"{d_total:,}")

st.write(
    "In this PyTorch version, both models have trainable weights by default because their layers are defined inside `nn.Module` and their parameters are passed into Adam optimizers. During the generator step, discriminator gradients are temporarily disabled so only the generator updates."
)

train_col, sample_col = st.columns([1.2, 1])

with train_col:
    st.subheader("Train the GAN")
    if st.button("Start training"):
        dataloader = make_dataloader(batch_size=batch_size, root=data_dir)
        progress_bar = st.progress(0)
        status = st.empty()
        rows: list[dict] = []

        def on_progress(update: dict) -> None:
            rows.append(update)
            progress_bar.progress(update["epoch"] / epochs)
            status.write(
                f"Epoch {update['epoch']}/{epochs} | D loss: {update['d_loss']:.4f} | D acc: {update['d_acc']:.4f} | G loss: {update['g_loss']:.4f}"
            )

        history = train_gan(
            components,
            dataloader,
            epochs=epochs,
            sample_dir=OUTPUT_DIR / "samples",
            checkpoint_dir=OUTPUT_DIR / "checkpoints",
            progress_callback=on_progress,
        )
        history_df = pd.DataFrame(rows)
        st.success("Training completed.")
        st.dataframe(history_df, use_container_width=True)

        fig = plt.figure(figsize=(8, 4))
        plt.plot(history_df["epoch"], history_df["d_loss"], label="Discriminator loss")
        plt.plot(history_df["epoch"], history_df["g_loss"], label="Generator loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with sample_col:
    st.subheader("Generate digits from saved model")
    if st.button("Load latest checkpoint and generate"):
        if not CHECKPOINT_PATH.exists():
            st.error("No checkpoint found yet. Train the model first.")
        else:
            generator = load_generator_from_checkpoint(CHECKPOINT_PATH, device=device)
            images = generate_images(generator, device=torch.device(device), n=16)
            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for ax, img in zip(axes.flatten(), images):
                ax.imshow(img.squeeze(0), cmap="gray")
                ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

samples_dir = OUTPUT_DIR / "samples"
if samples_dir.exists():
    pngs = sorted(samples_dir.glob("*.png"))
    if pngs:
        st.subheader("Saved sample grids")
        st.image([str(p) for p in pngs[-4:]], caption=[p.name for p in pngs[-4:]], width=180)
