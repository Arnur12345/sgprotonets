"""Entry point for evaluating a trained SGProtoNet checkpoint."""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.sgprotonet import SGProtoNet
from evaluation.evaluate import meta_test


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
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Build config from checkpoint, optionally overridden
    cfg = OmegaConf.create(ckpt["config"])
    if args.config:
        override_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, override_cfg)
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    seed_everything(cfg.seed)

    # Build and load model
    model = SGProtoNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    results = meta_test(model, cfg, device, split=args.split)

    print(f"\n{'='*50}")
    print(f"Accuracy: {results['mean_accuracy']:.4f} +/- {results['ci_95']:.4f}")
    print(f"{'='*50}")

    # Save results
    out_path = Path(args.checkpoint).parent / f"eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump({
            "mean_accuracy": results["mean_accuracy"],
            "ci_95": results["ci_95"],
            "per_class_acc": {str(k): v for k, v in results["per_class_acc"].items()},
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
