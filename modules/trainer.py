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

    def _to_device(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}

    def train_epoch(self, optimizer) -> dict[str, float]:
        self.model.train()

        totals = {
            "loss": torch.zeros((), device=self.device),
            "geographic": torch.zeros((), device=self.device),
            "contrastive": torch.zeros((), device=self.device),
        }
        batches = 0

        progress = tqdm(self.loader, desc="Training", dynamic_ncols=True)

        for image_inputs, text_inputs, labels, metadata, num_views in progress:
            image_inputs = self._to_device(image_inputs)
            text_inputs = self._to_device(text_inputs)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = len(metadata)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                geo_logits, image_embeddings, text_embeddings = self.model(
                    image_inputs,
                    text_inputs,
                    batch_size,
                    num_views,
                )
                loss, parts = self.loss_fn(
                    geo_logits,
                    labels,
                    image_embeddings,
                    text_embeddings,
                    self.model.logit_scale,
                    self.model.logit_bias,
                )

            loss.backward()
            optimizer.step()

            totals["loss"] += loss.detach()
            totals["geographic"] += parts["geographic"]
            totals["contrastive"] += parts["contrastive"]
            batches += 1

            progress.set_postfix(
                loss=f"{loss.item():.3f}",
                geo=f"{parts['geographic'].item():.3f}",
                contrast=f"{parts['contrastive'].item():.3f}",
            )

        denominator = max(batches, 1)
        return {name: (value / denominator).item() for name, value in totals.items()}

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
