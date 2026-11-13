"""Look inside a Vision Transformer (ViT).

The script downloads a small pretrained ViT (DeiT-Tiny, about 23 MB) and runs it
on one image. It saves five figures that show each step:

    1_patches.png          the image cut into 16 x 16 pixel patches
    2_patch_embedding.png  one patch and the vector of numbers the model makes from it
    3_embeddings_pca.png   all patch vectors shown as colours, first and last layer
    4_attention.png        which patches the model uses for its decision
    5_prediction.png       the five most likely ImageNet classes

Usage (from the repository root):
    python demo/vit_demo.py
    python demo/vit_demo.py --image demo/images/dogs.jpg
    python demo/vit_demo.py --image path/to/your_photo.jpg --patch 90
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed (works on servers too)

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # hide library warnings that are not useful here
hf_logging.disable_progress_bar()

DEFAULT_MODEL = "facebook/deit-tiny-patch16-224"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show how a Vision Transformer sees an image.")
    parser.add_argument("--image", default="demo/images/cat.jpg", help="path to an image file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name")
    parser.add_argument("--patch", type=int, default=None,
                        help="index of the patch to inspect (0 = top left); default: image centre")
    parser.add_argument("--out", default="outputs", help="folder for the figures")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def step(number: int, text: str) -> None:
    print(f"\n[Step {number}] {text}")


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_display(pixel_values: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    """Undo the normalisation so we can show exactly what the model received."""
    img = pixel_values[0].cpu().permute(1, 2, 0).numpy()
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)


def pca_rgb(tokens: np.ndarray, grid: int) -> np.ndarray:
    """Project patch vectors to 3 numbers each (PCA) and show them as RGB colours."""
    centred = tokens - tokens.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    rgb = centred @ vt[:3].T
    rgb = (rgb - rgb.min(axis=0)) / (rgb.max(axis=0) - rgb.min(axis=0) + 1e-8)
    return rgb.reshape(grid, grid, 3)


def attention_rollout(attentions: tuple[torch.Tensor, ...]) -> np.ndarray:
    """Follow attention through all layers (Abnar and Zuidema, 2020).

    Each layer mixes tokens. Multiplying the layer attention matrices shows how much
    each input patch finally flows into the class token.
    """
    tokens = attentions[0].shape[-1]
    rollout = torch.eye(tokens)
    for layer in attentions:
        a = layer[0].mean(dim=0).cpu()          # average over heads: (tokens, tokens)
        a = a + torch.eye(tokens)               # keep the skip connection
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout
    return rollout[0, 1:].numpy()               # class token row, patch columns


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    image_path = Path(args.image)
    out_dir = Path(args.out) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    step(1, f"Load the image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"  original size: {image.width} x {image.height} pixels")

    step(2, f"Load the model '{args.model}' (downloaded once, then cached)")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = ViTForImageClassification.from_pretrained(args.model, attn_implementation="eager")
    model.to(device).eval()
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ready in {time.time() - t0:.1f} s, running on: {device}")
    print(f"  parameters: {n_params / 1e6:.1f} million")
    print(f"  layers: {cfg.num_hidden_layers}, attention heads: {cfg.num_attention_heads}, "
          f"vector size: {cfg.hidden_size}")

    step(3, "Prepare the image (resize, crop, normalise)")
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]
    size = pixel_values.shape[-1]
    p = cfg.patch_size
    grid = size // p
    n_patches = grid * grid
    print(f"  model input shape: {tuple(pixel_values.shape)}  (batch, colours, height, width)")

    step(4, "Cut the image into patches")
    print(f"  {size} / {p} = {grid} patches per side, {grid} x {grid} = {n_patches} patches")
    print(f"  each patch has {p} x {p} x 3 = {p * p * 3} pixel values")
    shown = to_display(pixel_values, processor.image_mean, processor.image_std)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(shown)
    for k in range(1, grid):
        axes[0].axhline(k * p - 0.5, color="white", lw=0.6)
        axes[0].axvline(k * p - 0.5, color="white", lw=0.6)
    axes[0].set_title(f"Model input {size}x{size}, cut into {n_patches} patches")
    gap = 3
    exploded = np.ones((grid * (p + gap), grid * (p + gap), 3))
    for r in range(grid):
        for c in range(grid):
            y, x = r * (p + gap), c * (p + gap)
            exploded[y:y + p, x:x + p] = shown[r * p:(r + 1) * p, c * p:(c + 1) * p]
    axes[1].imshow(exploded)
    axes[1].set_title("The same patches, pulled apart (a sequence of 'tokens')")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "1_patches.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    step(5, "Run the transformer")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(pixel_values, output_hidden_states=True, output_attentions=True)
    print(f"  forward pass took {(time.time() - t0) * 1000:.0f} ms")
    hidden = [h[0].cpu().numpy() for h in outputs.hidden_states]  # each: (tokens, hidden_size)
    print(f"  tokens per image: {hidden[0].shape[0]} = 1 class token + {n_patches} patch tokens")
    print(f"  each token is a vector of {hidden[0].shape[1]} numbers (the embedding)")
    print(f"  embeddings shape after each layer: {hidden[0].shape}, "
          f"saved for {len(hidden)} stages (input + {cfg.num_hidden_layers} layers)")

    idx = args.patch if args.patch is not None else (grid // 2) * grid + grid // 2
    if not 0 <= idx < n_patches:
        raise SystemExit(f"--patch must be between 0 and {n_patches - 1}")
    row, col = divmod(idx, grid)
    vec = hidden[0][1 + idx]  # +1 skips the class token
    print(f"  patch {idx} (row {row}, column {col}): first 8 numbers of its embedding:")
    print("   ", np.array2string(vec[:8], precision=3, suppress_small=True))

    fig = plt.figure(figsize=(12, 4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(shown)
    ax0.add_patch(plt.Rectangle((col * p - 0.5, row * p - 0.5), p, p, fill=False, ec="red", lw=2))
    ax0.set_title(f"Patch {idx} (row {row}, col {col})")
    ax0.axis("off")
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(shown[row * p:(row + 1) * p, col * p:(col + 1) * p], interpolation="nearest")
    ax1.set_title(f"The patch: {p}x{p}x3 = {p * p * 3} numbers")
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.bar(np.arange(vec.size), vec, width=1.0, color="#0072B2")
    ax2.set_title(f"Its embedding: {vec.size} numbers")
    ax2.set_xlabel("dimension")
    ax2.set_ylabel("value")
    fig.tight_layout()
    fig.savefig(out_dir / "2_patch_embedding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    axes[0].imshow(shown)
    axes[0].set_title("Input")
    axes[1].imshow(pca_rgb(hidden[0][1:], grid), interpolation="nearest")
    axes[1].set_title("Before layer 1: mostly position")
    axes[2].imshow(pca_rgb(hidden[-1][1:], grid), interpolation="nearest")
    axes[2].set_title(f"After layer {cfg.num_hidden_layers}: image content")
    for ax in axes:
        ax.axis("off")
    fig.text(0.5, 0.02, "Each square is one patch. Similar colours = similar vectors (3-component PCA).",
             ha="center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / "3_embeddings_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    step(6, "Read the attention")
    attn = outputs.attentions[-1][0].cpu().numpy()  # (heads, tokens, tokens)
    print(f"  attention shape in one layer: {attn.shape}  (heads, tokens, tokens)")
    print("  each row says how much one token looks at every other token")
    maps = [attn[h, 0, 1:].reshape(grid, grid) for h in range(attn.shape[0])]
    names = [f"Last layer, head {h + 1}" for h in range(attn.shape[0])]
    maps.append(attention_rollout(outputs.attentions).reshape(grid, grid))
    names.append("All layers combined")
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(3.2 * (len(maps) + 1), 3.6))
    axes[0].imshow(shown)
    axes[0].set_title("Input")
    for ax, amap, name in zip(axes[1:], maps, names):
        ax.imshow(shown)
        ax.imshow(amap, cmap="inferno", alpha=0.6, extent=(0, size, size, 0), interpolation="bilinear")
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "4_attention.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    step(7, "Prediction")
    probs = outputs.logits[0].softmax(dim=-1).cpu()
    top = probs.topk(5)
    labels = [cfg.id2label[i].split(",")[0] for i in top.indices.tolist()]
    for label, prob in zip(labels, top.values.tolist()):
        print(f"  {prob * 100:5.1f} %  {label}")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(labels[::-1], (top.values * 100).tolist()[::-1], color="#009E73")
    ax.set_xlabel("probability (%)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out_dir / "5_prediction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDone. Open the figures in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
