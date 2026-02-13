"""Entry point for training SGProtoNet."""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.trainer import Trainer



def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SGProtoNet")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format (e.g. episode.k_shot=5)",
    )
    args = parser.parse_args()

    # Load config: merge default + experiment + CLI overrides
    default_cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    )
    exp_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(default_cfg, exp_cfg, cli_cfg)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    seed_everything(cfg.seed)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
