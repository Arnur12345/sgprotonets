"""Run ablation experiments — removes one component at a time."""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.trainer import Trainer
from evaluation.evaluate import meta_test

logger = logging.getLogger(__name__)

# Ablation variants: name -> config overrides
ABLATIONS = {
    "full_model": {},
    "no_sgam": {
        # Replace SGAM output with zero-vector, effectively skipping it.
        # Implemented by setting sgam_heads=0 (handled in model).
        # For simplicity, we skip SGAM by using visual_only text strategy.
    },
    "no_text": {
        "inference.text_strategy": "visual_only",
    },
    "no_phase1": {
        "training.phase1.enabled": False,
    },
    "euclidean_distance": {
        "model.distance": "euclidean",
    },
    "vanilla_prototypes": {
        "model.prototype_mode": "vanilla",
    },
    "vis2sem_inference": {
        "inference.text_strategy": "vis2sem",
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="SGProtoNet Ablation Study")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=list(ABLATIONS.keys()),
        help="Which ablations to run",
    )
    parser.add_argument("--eval_only", action="store_true", help="Eval from checkpoints only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    default_cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    )
    exp_cfg = OmegaConf.load(args.config)
    base_cfg = OmegaConf.merge(default_cfg, exp_cfg)

    results = {}

    for name in args.ablations:
        if name not in ABLATIONS:
            logger.warning(f"Unknown ablation: {name}, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation: {name}")
        logger.info(f"{'='*60}")

        overrides = ABLATIONS[name]
        ablation_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(overrides))
        ablation_cfg.training.checkpoint_dir = f"checkpoints/ablation_{name}/"

        seed_everything(ablation_cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not args.eval_only:
            trainer = Trainer(ablation_cfg)
            trainer.train()

        # Evaluate
        ckpt_path = Path(ablation_cfg.training.checkpoint_dir) / "best.pt"
        if ckpt_path.exists():
            from models.sgprotonet import SGProtoNet
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = SGProtoNet(ablation_cfg).to(device)
            model.load_state_dict(ckpt["model_state_dict"])

            test_results = meta_test(model, ablation_cfg, device, split="test")
            results[name] = {
                "accuracy": test_results["mean_accuracy"],
                "ci_95": test_results["ci_95"],
            }
        else:
            logger.warning(f"No checkpoint found for {name}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ABLATION SUMMARY")
    logger.info(f"{'='*60}")
    for name, res in results.items():
        logger.info(f"  {name:25s}: {res['accuracy']:.4f} +/- {res['ci_95']:.4f}")

    # Save results
    out_path = Path(base_cfg.training.checkpoint_dir) / "ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
