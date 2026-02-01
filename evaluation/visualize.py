"""SGAM attention map visualization."""

import math

import torch
import numpy as np
from PIL import Image


@torch.no_grad()
def get_attention_map(
    model,
    image: torch.Tensor,
    text: str,
    device: torch.device,
    image_size: int = 224,
) -> np.ndarray:
    """Extract SGAM attention map for a single image-text pair.

    Args:
        model: SGProtoNet model.
        image: Preprocessed image tensor, shape (3, H, W).
        text: Report text string.
        device: Target device.
        image_size: Original image size for reshaping attention.

    Returns:
        Attention map as numpy array, shape (H_patches, W_patches).
    """
    model.eval()
    image = image.unsqueeze(0).to(device)

    out = model.encode_multimodal(image, [text])
    attn_weights = out["attn_weights"].squeeze(0).cpu().numpy()  # (num_patches,)

    # Reshape to spatial grid
    num_patches = attn_weights.shape[0]
    grid_size = int(math.sqrt(num_patches))
    attn_map = attn_weights.reshape(grid_size, grid_size)

    return attn_map


def overlay_attention(
    image: Image.Image,
    attn_map: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay attention heatmap on original image.

    Args:
        image: Original PIL image.
        attn_map: Attention map, shape (H_grid, W_grid).
        alpha: Blending factor for the heatmap.

    Returns:
        Blended PIL image.
    """
    import matplotlib.cm as cm

    # Normalize attention to [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Resize to image dimensions
    w, h = image.size
    attn_resized = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    # Apply colormap
    heatmap = cm.jet(attn_resized)[:, :, :3]  # (H, W, 3)
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend
    image_arr = np.array(image)
    blended = (alpha * heatmap + (1 - alpha) * image_arr).astype(np.uint8)

    return Image.fromarray(blended)
