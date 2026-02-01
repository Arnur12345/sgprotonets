"""Learning rate scheduling with warmup."""

import math

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: DictConfig,
    total_epochs: int,
) -> LambdaLR:
    """Build a learning rate scheduler with warmup.

    Args:
        optimizer: PyTorch optimizer.
        cfg: training.scheduler config section.
        total_epochs: Total number of epochs.

    Returns:
        LambdaLR scheduler.
    """
    warmup_epochs = cfg.warmup_epochs
    min_lr = cfg.min_lr
    base_lr = optimizer.param_groups[0]["lr"]

    if cfg.name == "cosine":
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr / base_lr, cosine_decay)
    elif cfg.name == "step":
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            # Step decay at 60% and 80% of total epochs
            if epoch >= int(0.8 * total_epochs):
                return max(min_lr / base_lr, 0.01)
            elif epoch >= int(0.6 * total_epochs):
                return max(min_lr / base_lr, 0.1)
            return 1.0
    else:
        raise ValueError(f"Unknown scheduler: {cfg.name}")

    return LambdaLR(optimizer, lr_lambda)
