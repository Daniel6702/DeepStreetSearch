"""Optimizer learning-rate schedules, indexed by optimizer update count."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, mode: str) -> float:
    if not 0 <= warmup_steps <= total_steps or total_steps <= 0:
        raise ValueError("Require 0 <= warmup_steps <= total_steps and total_steps > 0")
    if mode not in ("cosine", "none"):
        raise ValueError("lr scheduler must be 'cosine' or 'none'")

    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if mode == "none":
        return 1.0
    if warmup_steps == total_steps:
        return 1.0

    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_scheduler(optimizer, total_steps: int, warmup_ratio: float, mode: str):
    if not 0 <= warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be between 0 and 1")
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(step, total_steps, warmup_steps, mode),
    ), warmup_steps
