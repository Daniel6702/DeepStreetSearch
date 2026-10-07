from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, loader, loss_fn, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loader = loader
        self.loss_fn = loss_fn.to(self.device)

    def _to_device(self, image_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in image_inputs.items()
        }

    def train_epoch(self, optimizer) -> float:
        self.model.train()
        running_loss = torch.zeros((), device=self.device)
        batches = 0

        #training loop

        return (running_loss / max(batches, 1)).item()

    def save_checkpoint(self, optimizer, epoch: int, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            path,
        )
