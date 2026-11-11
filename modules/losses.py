from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_geogrid import AdaptiveGeoGrid
from config import NUM_VIEWS


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
    """SigLIP pairwise sigmoid objective with multiple texts per image.

    Optionally ignore other crops of the positive panorama, since they may
    show overlapping content and should not automatically be negatives.
    """

    def __init__(self, revised_contrastive_positives: bool = False, num_views: int = NUM_VIEWS):
        super().__init__()
        if num_views < 1:
            raise ValueError("num_views must be positive")
        self.revised_contrastive_positives = revised_contrastive_positives
        self.num_views = num_views

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        positive_image_indices: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor,
    ) -> torch.Tensor:
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError("Image and text embeddings must both have shape [items, dim]")
        if image_embeddings.shape[1] != text_embeddings.shape[1]:
            raise ValueError(
                "Image and text embedding dimensions must match, "
                f"got {image_embeddings.shape[1]} and {text_embeddings.shape[1]}"
            )
        if positive_image_indices.shape != (text_embeddings.shape[0],):
            raise ValueError(
                "positive_image_indices must contain one image index per text embedding, "
                f"got {tuple(positive_image_indices.shape)} for {text_embeddings.shape[0]} texts"
            )

        positive_image_indices = positive_image_indices.to(image_embeddings.device, non_blocking=True)
        if positive_image_indices.numel() and (
            positive_image_indices.min() < 0 or positive_image_indices.max() >= image_embeddings.shape[0]
        ):
            raise ValueError("positive_image_indices contains an out-of-range image index")

        logits = text_embeddings @ image_embeddings.T
        logits = logits * logit_scale.exp() + logit_bias

        signs = -torch.ones_like(logits) 
        text_indices = torch.arange(text_embeddings.shape[0], device=logits.device)
        signs[text_indices, positive_image_indices] = 1.0

        pair_losses = -F.logsigmoid(signs * logits)
        if self.revised_contrastive_positives:
            if image_embeddings.shape[0] % self.num_views:
                raise ValueError("Image embeddings must contain complete panoramas")
            # View indices are panorama-major, with num_views consecutive crops
            # for each panorama. Only the description's own crop is positive;
            # sibling crops are ignored. Crops from other panoramas are negative.
            image_indices = torch.arange(image_embeddings.shape[0], device=logits.device)
            same_panorama = (
                positive_image_indices[:, None] // self.num_views
                == image_indices[None, :] // self.num_views
            )
            same_panorama[text_indices, positive_image_indices] = False
            pair_losses = pair_losses.masked_fill(same_panorama, 0.0)

        return pair_losses.sum(dim=-1).mean()


class DeepStreetSearchLoss(nn.Module):
    def __init__(
        self,
        grid: AdaptiveGeoGrid,
        hierarchy_weights: list[float] | None = None,
        geographic_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        revised_contrastive_positives: bool = False,
        panorama_contrastive_weight: float = 0.0,
        panorama_contrastive_aggregation: str = "mean",
        semantic_token_loss_weight: float = 0.0,
    ):
        super().__init__()

        if min(geographic_weight, contrastive_weight, panorama_contrastive_weight,
               semantic_token_loss_weight) < 0:
            raise ValueError("Loss weights must be non-negative")
        if not any((geographic_weight, contrastive_weight, panorama_contrastive_weight,
                    semantic_token_loss_weight)):
            raise ValueError("At least one loss weight must be positive")
        if panorama_contrastive_aggregation not in {"mean", "max"}:
            raise ValueError("panorama_contrastive_aggregation must be 'mean' or 'max'")

        self.geographic = HierarchicalGeographicLoss(grid, hierarchy_weights)
        self.contrastive = SiglipContrastiveLoss(
            revised_contrastive_positives=revised_contrastive_positives
        )
        # A text can match one of six views while also being positive for the
        # whole panorama. Unlike the crop objective, every other panorama is a
        # negative; there are no sibling-view pairs to mask at this level.
        self.panorama_contrastive = SiglipContrastiveLoss(num_views=1)
        self.geographic_weight = geographic_weight
        self.contrastive_weight = contrastive_weight
        self.panorama_contrastive_weight = panorama_contrastive_weight
        self.panorama_contrastive_aggregation = panorama_contrastive_aggregation
        self.semantic_token_loss_weight = semantic_token_loss_weight

    @staticmethod
    def _multi_vector_panorama_loss(
        vectors: torch.Tensor,
        text_embeddings: torch.Tensor,
        positive_image_indices: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor,
    ) -> torch.Tensor:
        """One positive panorama per crop caption; MaxSim across its vectors."""
        if vectors.ndim != 3 or vectors.shape[1] < 1:
            raise ValueError("Panorama vectors must have shape [panoramas, vectors, dim]")
        if text_embeddings.ndim != 2 or text_embeddings.shape[1] != vectors.shape[2]:
            raise ValueError("Panorama and text vector dimensions differ")
        if positive_image_indices.shape != (text_embeddings.shape[0],):
            raise ValueError("Expected one positive crop index per text embedding")
        panorama_positives = torch.div(positive_image_indices, NUM_VIEWS, rounding_mode="floor")
        if panorama_positives.numel() and (
            panorama_positives.min() < 0 or panorama_positives.max() >= vectors.shape[0]
        ):
            raise ValueError("Positive panorama index outside batch")
        # Normalize explicitly to avoid shortcuts through token vector norms.
        scores = torch.einsum("td,bkd->tbk", text_embeddings, F.normalize(vectors, dim=-1)).amax(-1)
        logits = scores * logit_scale.exp() + logit_bias
        signs = -torch.ones_like(logits)
        signs[torch.arange(len(text_embeddings), device=logits.device),
              panorama_positives.to(logits.device)] = 1.0
        return -F.logsigmoid(signs * logits).sum(dim=-1).mean()

    def forward(
        self,
        geo_logits: torch.Tensor,
        labels: torch.Tensor,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        positive_image_indices: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor,
        semantic_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        geographic = self.geographic(geo_logits, labels)
        if self.contrastive_weight > 0:
            contrastive = self.contrastive(
                image_embeddings, text_embeddings, positive_image_indices, logit_scale, logit_bias
            )
        else:
            # Do not create zero gradients for backbone parameters: AdamW would
            # otherwise still apply weight decay and change retrieval weights.
            contrastive = geographic.new_zeros(())

        if self.panorama_contrastive_weight > 0:
            if image_embeddings.ndim != 2 or image_embeddings.shape[0] % NUM_VIEWS:
                raise ValueError("Panorama contrastive loss requires complete view groups")
            panorama_positives = torch.div(
                positive_image_indices, NUM_VIEWS, rounding_mode="floor"
            )
            if self.panorama_contrastive_aggregation == "mean":
                # Preserve the existing optional mean objective unchanged.
                panorama_embeddings = F.normalize(
                    image_embeddings.reshape(-1, NUM_VIEWS, image_embeddings.shape[-1]).mean(dim=1),
                    dim=-1,
                )
                panorama_contrastive = self.panorama_contrastive(
                    panorama_embeddings, text_embeddings, panorama_positives,
                    logit_scale, logit_bias,
                )
            else:
                panorama_contrastive = self._multi_vector_panorama_loss(
                    image_embeddings.reshape(-1, NUM_VIEWS, image_embeddings.shape[-1]),
                    text_embeddings, positive_image_indices, logit_scale, logit_bias,
                )
        else:
            panorama_contrastive = geographic.new_zeros(())

        if self.semantic_token_loss_weight > 0:
            if semantic_tokens is None:
                raise ValueError("Semantic token loss requires semantic_tokens from the model")
            semantic_contrastive = self._multi_vector_panorama_loss(
                semantic_tokens, text_embeddings, positive_image_indices, logit_scale, logit_bias,
            )
        else:
            semantic_contrastive = geographic.new_zeros(())

        total = (self.geographic_weight * geographic
                 + self.contrastive_weight * contrastive
                 + self.panorama_contrastive_weight * panorama_contrastive
                 + self.semantic_token_loss_weight * semantic_contrastive)
        return total, {
            "geographic": geographic.detach(),
            "contrastive": contrastive.detach(),
            "panorama_contrastive": panorama_contrastive.detach(),
            "semantic_contrastive": semantic_contrastive.detach(),
        }
