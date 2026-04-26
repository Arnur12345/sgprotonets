"""Tier 2 sanity check: does the TRAINED SGProtoNet rely on report text?

Runs the same episodic eval used during validation, but under 4 ablations:
  A) Standard eval     — images, class anchors, no per-image text (current _validate).
  B) Multimodal+reports — images + real per-image reports for support and query.
  C) Text-only         — images ZEROED, real reports pass through.
  D) Visual-only       — images, text_strategy='visual_only' (no semantic input).

If C ≈ B and D ≪ B, the trained model is reading the answer from the report
instead of from the image.

Memory-safe: AMP autocast, num_workers=0, per-episode cache clearing, smaller
default q_query, and the loader is rebuilt fresh per condition.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.iu_xray import IUXRayDataset
from data.episode_sampler import EpisodeSampler, unpack_episode
from data.class_descriptions import CLASS_DESCRIPTIONS
from models.sgprotonet import SGProtoNet


@torch.no_grad()
def get_class_anchors(model: SGProtoNet, classes: list[str], device) -> torch.Tensor:
    descs = [CLASS_DESCRIPTIONS.get(c, "") for c in classes]
    s_cls, _ = model.semantic_encoder(descs, device)
    return model.semantic_proj(s_cls)


def make_loader(dataset, n_way, k_shot, q_query, num_episodes, seed):
    sampler = EpisodeSampler(
        dataset=dataset,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=num_episodes,
        seed=seed,
    )
    # num_workers=0 to avoid worker copies of the dataset/transforms in RAM
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)


@torch.no_grad()
def run_condition(
    model: SGProtoNet,
    dataset,
    n_way: int,
    k_shot: int,
    q_query: int,
    num_episodes: int,
    seed: int,
    class_anchors: torch.Tensor,
    device,
    condition: str,
    use_amp: bool,
) -> dict:
    """Run all episodes under one condition and return mean accuracy."""
    model.eval()
    accs = []
    loader = make_loader(dataset, n_way, k_shot, q_query, num_episodes, seed)

    for batch in loader:
        support, query, query_labels = unpack_episode(batch, n_way, k_shot, q_query)

        support_images = support["image"].to(device, non_blocking=False)
        query_images = query["image"].to(device, non_blocking=False)
        support_labels = support["label"].to(device)
        query_labels = query_labels.to(device)
        support_reports = support["report"]
        query_reports = query["report"]

        # Episode anchors for the n_way classes in this episode
        orig_labels = batch["label"][: n_way * k_shot]
        episode_class_indices = orig_labels[::k_shot].tolist()
        episode_anchors = class_anchors[episode_class_indices]

        if condition == "A_standard":
            sup_imgs = support_images
            qry_imgs = query_images
            sup_txt = None
            qry_txt = None
            anchors = episode_anchors
            strategy = "class_anchors"

        elif condition == "B_multimodal_reports":
            sup_imgs = support_images
            qry_imgs = query_images
            sup_txt = support_reports
            qry_txt = query_reports
            anchors = episode_anchors
            strategy = "class_anchors"

        elif condition == "C_text_only":
            sup_imgs = torch.zeros_like(support_images)
            qry_imgs = torch.zeros_like(query_images)
            sup_txt = support_reports
            qry_txt = query_reports
            anchors = None
            strategy = "class_anchors"

        elif condition == "D_visual_only":
            sup_imgs = support_images
            qry_imgs = query_images
            sup_txt = None
            qry_txt = None
            anchors = None
            strategy = "visual_only"
        else:
            raise ValueError(condition)

        with torch.cuda.amp.autocast(enabled=use_amp):
            ep_out = model.forward_episode(
                support_images=sup_imgs,
                support_texts=sup_txt,
                support_labels=support_labels,
                query_images=qry_imgs,
                query_texts=qry_txt,
                n_way=n_way,
                class_semantic_embeds=anchors,
                text_strategy=strategy,
            )
            preds = ep_out["logits"].argmax(dim=-1)

        accs.append((preds == query_labels).float().mean().item())

        # Free per-episode tensors
        del support_images, query_images, sup_imgs, qry_imgs, ep_out, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "n_episodes": len(accs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Device: {device} (AMP: {use_amp})")
    print(f"Checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    print(f"Trained for {ckpt.get('epoch', '?')} epochs, "
          f"val acc at save: {ckpt.get('accuracy', '?')}")

    model = SGProtoNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    split_classes = list(cfg.data[f"{args.split}_classes"])
    print(f"\nSplit '{args.split}' classes: {split_classes}")

    dataset = IUXRayDataset(
        data_dir=args.data_dir,
        split_classes=split_classes,
        image_size=cfg.data.image_size,
        is_train=False,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataset.classes)} classes")

    n_way = min(args.n_way, len(dataset.classes))
    class_anchors = get_class_anchors(model, dataset.classes, device)

    print(
        f"\nRunning {args.num_episodes} episodes "
        f"({n_way}-way, {args.k_shot}-shot, {args.q_query}-query) per condition...\n"
    )

    results = {}
    for cond in ["A_standard", "B_multimodal_reports", "C_text_only", "D_visual_only"]:
        print(f"  [{cond}] running...")
        res = run_condition(
            model=model,
            dataset=dataset,
            n_way=n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            num_episodes=args.num_episodes,
            seed=args.seed,
            class_anchors=class_anchors,
            device=device,
            condition=cond,
            use_amp=use_amp,
        )
        results[cond] = res
        print(f"    acc = {res['mean']:.4f} ± {res['std']:.4f}  ({res['n_episodes']} eps)")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    chance = 1.0 / n_way
    a = results["A_standard"]["mean"]
    b = results["B_multimodal_reports"]["mean"]
    c = results["C_text_only"]["mean"]
    d = results["D_visual_only"]["mean"]

    print("\n" + "=" * 70)
    print("TIER 2 — Trained-model leakage probe")
    print("=" * 70)
    print(f"  Chance ({n_way}-way):                       {chance:.4f}")
    print(f"  A) Standard (anchors, no text):       {a:.4f}")
    print(f"  B) Multimodal w/ real reports:        {b:.4f}")
    print(f"  C) Text-only (images zeroed):         {c:.4f}")
    print(f"  D) Visual-only:                       {d:.4f}")
    print("=" * 70)

    print("\nInterpretation:")
    if c > chance + 0.20:
        print(f"  ! Text-only ({c:.3f}) >> chance ({chance:.3f}) — "
              "model classifies from REPORTS ALONE.")
    if c >= b - 0.05:
        print(f"  ! Text-only ≈ multimodal ({c:.3f} vs {b:.3f}) — visual stream is dispensable.")
    if d < a - 0.05:
        print(f"  ! Visual-only ({d:.3f}) << standard ({a:.3f}) — visual encoder under-trained.")
    if b > a + 0.05:
        print(f"  ! Reports boost over standard ({b:.3f} vs {a:.3f}) — "
              "training-time reports left a usable shortcut.")


if __name__ == "__main__":
    main()
