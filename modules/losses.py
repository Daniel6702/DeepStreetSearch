from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_geogrid import AdaptiveGeoGrid


class HierarchicalGeographicLoss(nn.Module):
    """DeepGeoGuesser-style fine-to-coarse geographic classification loss."""

    def __init__(self, grid: AdaptiveGeoGrid, weights: list[float] | None = None):
        super().__init__()

        if weights is None:
            weights = [1.0] * grid.n_levels

        if len(weights) != grid.n_levels:
            raise ValueError(f"Expected {grid.n_levels} hierarchy weights, got {len(weights)}")

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", weights_tensor / weights_tensor.sum())

        hierarchy = grid.hierarchy_table().to_numpy(dtype="int64")
        self.register_buffer("hierarchy", torch.tensor(hierarchy, dtype=torch.long))
        self.n_classes = grid.n_classes
        self.n_levels = grid.n_levels

    def forward(self, fine_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(fine_logits.device, non_blocking=True)

        valid = labels[:, 0] >= 0
        if not valid.any():
            raise ValueError("Batch contains no coordinates inside the geographic grid")

        fine_logits = fine_logits[valid]
        labels = labels[valid]

        total_loss = self.weights[0] * F.cross_entropy(fine_logits, labels[:, 0])
        fine_probs = F.softmax(fine_logits, dim=1)

        for level in range(1, self.n_levels):
            parent = self.hierarchy[:, level]
            num_classes = self.n_classes[level]

            coarse_probs = fine_probs.new_zeros(fine_probs.shape[0], num_classes)
            coarse_probs.scatter_add_(1, parent.unsqueeze(0).expand(fine_probs.shape[0], -1), fine_probs)

            total_loss = total_loss + self.weights[level] * F.nll_loss(
                torch.log(coarse_probs.clamp_min(1e-12)),
                labels[:, level],
            )

        return total_loss


class SiglipContrastiveLoss(nn.Module):
    """Native SigLIP pairwise sigmoid contrastive objective."""

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor,
    ) -> torch.Tensor:
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError(
                "Image and text embeddings must have matching [batch, dim] shapes, "
                f"got {tuple(image_embeddings.shape)} and {tuple(text_embeddings.shape)}"
            )

        logits = text_embeddings @ image_embeddings.T
        logits = logits * logit_scale.exp() + logit_bias

        signs = -torch.ones_like(logits)
        signs.fill_diagonal_(1.0)

        return -F.logsigmoid(signs * logits).sum(dim=-1).mean()


class DeepStreetSearchLoss(nn.Module):
    def __init__(
        self,
        grid: AdaptiveGeoGrid,
        hierarchy_weights: list[float] | None = None,
        geographic_weight: float = 1.0,
        contrastive_weight: float = 1.0,
    ):
        super().__init__()

        if geographic_weight < 0 or contrastive_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if geographic_weight == 0 and contrastive_weight == 0:
            raise ValueError("At least one loss weight must be positive")

        self.geographic = HierarchicalGeographicLoss(grid, hierarchy_weights)
        self.contrastive = SiglipContrastiveLoss()
        self.geographic_weight = geographic_weight
        self.contrastive_weight = contrastive_weight

    def forward(
        self,
        geo_logits: torch.Tensor,
        labels: torch.Tensor,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        geographic = self.geographic(geo_logits, labels)
        contrastive = self.contrastive(image_embeddings, text_embeddings, logit_scale, logit_bias)

        total = self.geographic_weight * geographic + self.contrastive_weight * contrastive
        return total, {
            "geographic": geographic.detach(),
            "contrastive": contrastive.detach(),
        }
