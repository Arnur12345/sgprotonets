"""Entry point for evaluating a trained SGProtoNet checkpoint."""

import argparse
import gc
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.sgprotonet import SGProtoNet
from evaluation.evaluate import meta_test, meta_test_multilabel


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SGProtoNet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, default=None, help="Override config (optional)")
    parser.add_argument("--split", type=str, default="val", choices=["test", "val"])
    parser.add_argument("--device", type=str, default=None, help="Force device: 'cpu' or 'cuda'")
    parser.add_argument("--num-episodes", type=int, default=None, help="Override num_eval_episodes")
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Device selection — default to CPU to avoid GPU OOM on small machines
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Build config from checkpoint, optionally overridden
    cfg = OmegaConf.create(ckpt["config"])
    if args.config:
        override_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, override_cfg)
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Override num_eval_episodes if specified
    if args.num_episodes is not None:
        cfg.inference.num_eval_episodes = args.num_episodes

    seed_everything(cfg.seed)

    # Build and load model
    model = SGProtoNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt  # Free checkpoint memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Model loaded from {args.checkpoint}")
    logger.info(f"Evaluating on split='{args.split}' with {cfg.inference.num_eval_episodes} episodes")

    # Choose evaluation mode
    is_multilabel = cfg.get("multilabel", {}).get("enabled", False)

    if is_multilabel:
        logger.info("Running multi-label evaluation (F1, AUC-ROC, exact match)")
        results = meta_test_multilabel(model, cfg, device, split=args.split)

        print(f"\n{'='*60}")
        print(f"  Multi-Label Evaluation Results ({args.split})")
        print(f"{'='*60}")
        print(f"  Mean F1 (per-episode):   {results['mean_f1']:.4f} +/- {results['ci_f1']:.4f}")
        print(f"  Exact Match:             {results['mean_exact_match']:.4f} +/- {results['ci_exact_match']:.4f}")
        print(f"{'─'*60}")
        print(f"  Micro F1:                {results['micro_f1']:.4f}")
        print(f"  Macro F1:                {results['macro_f1']:.4f}")
        print(f"  Micro Precision:         {results['micro_precision']:.4f}")
        print(f"  Micro Recall:            {results['micro_recall']:.4f}")
        print(f"{'─'*60}")
        print(f"  Micro AUC-ROC:           {results['micro_auc']:.4f}")
        print(f"  Macro AUC-ROC:           {results['macro_auc']:.4f}")
        print(f"{'─'*60}")

        # Per-label breakdown
        if results.get("per_label_metrics"):
            print(f"  Per-Label Breakdown:")
            for label, metrics in results["per_label_metrics"].items():
                auc = results.get("per_label_auc", {}).get(label, None)
                auc_str = f"{auc:.4f}" if auc is not None else "N/A"
                print(
                    f"    {label:>25s}:  F1={metrics['f1']:.4f}  "
                    f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
                    f"AUC={auc_str}  (n={metrics['support']})"
                )
        print(f"{'='*60}")

        # Save results
        save_results = {
            "split": args.split,
            "num_episodes": cfg.inference.num_eval_episodes,
            "mean_f1": results["mean_f1"],
            "ci_f1": results["ci_f1"],
            "mean_exact_match": results["mean_exact_match"],
            "ci_exact_match": results["ci_exact_match"],
            "micro_f1": results["micro_f1"],
            "macro_f1": results["macro_f1"],
            "micro_precision": results["micro_precision"],
            "micro_recall": results["micro_recall"],
            "micro_auc": results["micro_auc"],
            "macro_auc": results["macro_auc"],
            "per_label_metrics": results.get("per_label_metrics", {}),
            "per_label_auc": {
                str(k): v for k, v in results.get("per_label_auc", {}).items()
            },
        }
    else:
        logger.info("Running single-label evaluation (accuracy)")
        results = meta_test(model, cfg, device, split=args.split)

        print(f"\n{'='*50}")
        print(f"Accuracy: {results['mean_accuracy']:.4f} +/- {results['ci_95']:.4f}")
        print(f"{'='*50}")

        save_results = {
            "split": args.split,
            "mean_accuracy": results["mean_accuracy"],
            "ci_95": results["ci_95"],
            "per_class_acc": {str(k): v for k, v in results["per_class_acc"].items()},
        }

    # Save results to JSON
    out_path = Path(args.checkpoint).parent / f"eval_{args.split}_multilabel.json" if is_multilabel else Path(args.checkpoint).parent / f"eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
