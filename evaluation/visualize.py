"""SGAM attention map and t-SNE embedding visualization."""

from __future__ import annotations

import gc
import logging
import math
from pathlib import Path

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


# =============================================================================
# t-SNE Embedding Visualization
# =============================================================================

logger = logging.getLogger(__name__)

# Human-readable clinical names for each pathology class
CLINICAL_NAMES: dict[str, str] = {
    "effusion": "Effusion",
    "cardiomegaly": "Cardiomegaly",
    "infiltrate": "Infiltrate",
    "mass": "Mass",
    "nodule": "Nodule",
    "pneumonia": "Pneumonia",
    "atelectasis": "Atelectasis",
    "consolidation": "Consolidation",
    "pneumothorax": "Pneumothorax",
    "edema": "Edema",
    "emphysema": "Emphysema",
    "hernia": "Hernia",
    "fibrosis": "Fibrosis",
    "pleural_thickening": "Pleural Thickening",
    "no_finding": "No Finding",
}


def _dominant_label(
    multi_hot: np.ndarray,
    class_counts: np.ndarray,
) -> int:
    """Return the dominant (rarest) positive class index for a multi-hot vector.

    For multi-label samples, we pick the positive label with the fewest total
    occurrences in the dataset so that rare pathologies are visible in the plot.

    Args:
        multi_hot: Binary vector of shape (n_classes,).
        class_counts: Total positive count per class, shape (n_classes,).

    Returns:
        Integer class index of the dominant label.
    """
    positive_indices = np.where(multi_hot > 0)[0]
    if len(positive_indices) == 0:
        return 0
    # Pick the positive label with the smallest count (rarest)
    rarest_idx = positive_indices[np.argmin(class_counts[positive_indices])]
    return int(rarest_idx)


@torch.no_grad()
def extract_embeddings_multilabel(
    model,
    cfg,
    device: torch.device,
    split: str = "val",
    max_episodes: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract f_final embeddings and labels from multi-label episodes.

    Args:
        model: Trained SGProtoNet.
        cfg: Full OmegaConf configuration.
        device: Target device.
        split: Dataset split to use ("val" or "test").
        max_episodes: Maximum episodes to collect embeddings from.

    Returns:
        Tuple of (embeddings, dominant_labels, class_names) where:
            - embeddings: np.ndarray of shape (N, d_model)
            - dominant_labels: np.ndarray of shape (N,) with integer class indices
            - class_names: list of class name strings (sorted)
    """
    from omegaconf import DictConfig
    from data.iu_xray import IUXRayMultiLabelDataset
    from data.episode_sampler import (
        BinaryEpisodeSampler,
        unpack_binary_episode,
        binary_episode_collate_fn,
    )
    from data.class_descriptions import CLASS_DESCRIPTIONS

    model.eval()
    ml_cfg = cfg.multilabel

    classes = list(cfg.data.val_classes) if split == "val" else list(cfg.data.test_classes)
    if not classes:
        classes = list(cfg.data.train_classes)

    dataset = IUXRayMultiLabelDataset(
        data_dir=cfg.data.data_dir,
        split_classes=classes,
        image_size=cfg.data.image_size,
        is_train=False,
    )

    sampler = BinaryEpisodeSampler(
        dataset=dataset,
        n_labels=ml_cfg.n_labels,
        k_pos=ml_cfg.k_pos,
        k_neg=ml_cfg.k_neg,
        q_query=ml_cfg.q_query,
        num_episodes=max_episodes,
        min_positives=ml_cfg.get("min_positives", 3),
        min_negatives=ml_cfg.get("min_negatives", 10),
        seed=cfg.seed + 3000,
    )

    # Pre-compute class anchors
    descriptions = [CLASS_DESCRIPTIONS.get(c, "") for c in dataset.classes]
    s_cls, _ = model.semantic_encoder(descriptions, device)
    class_anchors = model.semantic_proj(s_cls)

    sample_counts = [len(dataset.positive_indices[c]) for c in range(dataset.num_classes)]

    # Compute global class counts for rarity-based dominant label selection
    class_counts = np.array(sample_counts, dtype=np.float32)

    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for ep_idx, episode_meta in enumerate(sampler):
        indices = episode_meta["indices"]
        samples = [dataset[i] for i in indices]
        batch = binary_episode_collate_fn(samples)

        episode_labels = episode_meta["episode_labels"]
        n_labels = episode_meta["n_labels"]
        k_pos = episode_meta["k_pos"]
        k_neg = episode_meta["k_neg"]
        q_query = episode_meta["q_query"]

        support_pos, support_neg, query, query_labels_tensor, _ = unpack_binary_episode(
            batch, episode_labels, n_labels, k_pos, k_neg, q_query
        )

        query_images = query["image"].to(device)
        query_texts = query["report"]
        episode_anchors = class_anchors[episode_labels]

        # Encode query set to get f_final embeddings
        query_strategy = "vis2sem" if cfg.inference.text_strategy == "class_anchors" else cfg.inference.text_strategy
        query_out = model.encode_multimodal(
            query_images, query_texts, query_strategy, None
        )

        embeddings = query_out["f_final"].cpu().numpy()  # (n_query, d_model)
        labels_np = query_labels_tensor.numpy()  # (n_query, n_labels)

        # Map episode-local labels back to global class indices
        # episode_labels contains the global class indices for this episode
        global_labels = np.zeros((labels_np.shape[0], dataset.num_classes), dtype=np.float32)
        for local_idx, global_idx in enumerate(episode_labels):
            global_labels[:, global_idx] = labels_np[:, local_idx]

        # Compute dominant label per sample (rarest positive class)
        dominant = np.array([
            _dominant_label(global_labels[i], class_counts)
            for i in range(global_labels.shape[0])
        ])

        all_embeddings.append(embeddings)
        all_labels.append(dominant)

        if (ep_idx + 1) % 20 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"  Extracted embeddings from {ep_idx + 1}/{max_episodes} episodes")

    embeddings_arr = np.concatenate(all_embeddings, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    return embeddings_arr, labels_arr, dataset.classes


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_dir: str = "results",
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
) -> Path:
    """Run t-SNE on embeddings and produce a publication-quality scatter plot.

    Args:
        embeddings: Feature matrix of shape (N, d_model).
        labels: Integer class labels of shape (N,).
        class_names: List of class name strings (index-aligned).
        output_dir: Directory to save plots.
        perplexity: t-SNE perplexity parameter.
        n_iter: Number of t-SNE iterations.
        random_state: Seed for reproducibility.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    logger.info(f"Running t-SNE on {embeddings.shape[0]} samples (d={embeddings.shape[1]})...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)  # (N, 2)

    # Identify which classes actually appear in the labels
    unique_labels = np.unique(labels)

    # Academic-style figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # High-contrast palette (tab10)
    cmap = plt.cm.get_cmap("tab10")

    for rank, cls_idx in enumerate(unique_labels):
        mask = labels == cls_idx
        raw_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
        display_name = CLINICAL_NAMES.get(raw_name, raw_name.replace("_", " ").title())
        color = cmap(rank % 10)

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color],
            label=f"{display_name} (n={mask.sum()})",
            s=30,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_title(
        "t-SNE Visualization of SGProtoNet Multimodal Embeddings",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)

    # Clean academic style: no grid, minimal spines
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="both", which="both", length=3, width=0.5)

    # Legend outside the plot area
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        markerscale=1.2,
    )

    plt.tight_layout()

    # Save to results/
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    png_path = out_path / "tsne_embeddings.png"
    pdf_path = out_path / "tsne_embeddings.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved t-SNE plot to {png_path} and {pdf_path}")
    return png_path


@torch.no_grad()
def generate_tsne_visualization(
    model,
    cfg,
    device: torch.device,
    split: str = "val",
    max_episodes: int = 50,
    output_dir: str = "results",
) -> Path:
    """End-to-end: extract embeddings, run t-SNE, and save the plot.

    Args:
        model: Trained SGProtoNet.
        cfg: Full OmegaConf configuration.
        device: Target device.
        split: Dataset split ("val" or "test").
        max_episodes: Number of episodes to extract embeddings from.
        output_dir: Directory to save output files.

    Returns:
        Path to the saved PNG file.
    """
    embeddings, labels, class_names = extract_embeddings_multilabel(
        model, cfg, device, split=split, max_episodes=max_episodes,
    )

    logger.info(
        f"Collected {embeddings.shape[0]} embeddings across "
        f"{len(np.unique(labels))} classes"
    )

    return plot_tsne(
        embeddings, labels, class_names, output_dir=output_dir,
    )
