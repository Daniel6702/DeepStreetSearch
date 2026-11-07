from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from adaptive_geogrid import AdaptiveGeoGrid
from config import SIGLIP_MODEL


def _pooled_features(output) -> torch.Tensor:
    """Support Transformers versions returning either a tensor or model output."""
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    raise TypeError(f"Unsupported SigLIP feature output type: {type(output)!r}")


class GeoSearchModel(nn.Module):
    """SigLIP retrieval encoder with an independently trained geographic branch."""

    def __init__(
        self,
        grid: AdaptiveGeoGrid,
        model_name: str = SIGLIP_MODEL,
        freeze_backbone: bool = False,
        attn_implementation: str = "sdpa",
        geo_head: str = "linear",
        geo_pooling: str = "mean",
        geo_view_dropout: float = 0.0,
        freeze_text_encoder: bool = False,
        geo_backbone_gradient_scale: float = 0.0,
    ):
        super().__init__()

        if geo_head not in {"linear", "mlp"}:
            raise ValueError("geo_head must be 'linear' or 'mlp'")
        if geo_pooling not in {"mean", "attention"}:
            raise ValueError("geo_pooling must be 'mean' or 'attention'")
        if not 0.0 <= geo_view_dropout < 1.0:
            raise ValueError("geo_view_dropout must be in [0, 1)")
        if not 0.0 <= geo_backbone_gradient_scale <= 1.0:
            raise ValueError("geo_backbone_gradient_scale must be in [0, 1]")

        self.geo_view_dropout = geo_view_dropout
        self.geo_pooling = geo_pooling
        self.freeze_text_encoder = freeze_text_encoder
        self.geo_backbone_gradient_scale = geo_backbone_gradient_scale

        self.backbone = AutoModel.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
        )

        if not hasattr(self.backbone, "get_image_features") or not hasattr(self.backbone, "get_text_features"):
            raise TypeError(f"{model_name!r} is not a compatible SigLIP-style image-text model")

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

            # Keep SigLIP's similarity calibration trainable.
            if hasattr(self.backbone, "logit_scale"):
                self.backbone.logit_scale.requires_grad = True
            if hasattr(self.backbone, "logit_bias"):
                self.backbone.logit_bias.requires_grad = True

        if freeze_text_encoder:
            text_model = getattr(self.backbone, "text_model", None)
            if text_model is None:
                raise TypeError(f"{model_name!r} has no separate text_model to freeze")
            for parameter in text_model.parameters():
                parameter.requires_grad = False

        vision_dim = self.backbone.config.vision_config.hidden_size
        self.geo_attention = (
            nn.Sequential(
                nn.LayerNorm(vision_dim),
                nn.Linear(vision_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            ) if geo_pooling == "attention" else None
        )
        if geo_head == "mlp":
            self.geo_head = nn.Sequential(
                nn.LayerNorm(vision_dim),
                nn.Linear(vision_dim, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, grid.n_classes[0]),
            )
        else:
            # Preserve legacy checkpoint parameter names for the linear model.
            self.geo_head = nn.Linear(vision_dim, grid.n_classes[0])

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen text tower must also stay in eval mode (no train-time dropout).
        if self.freeze_text_encoder:
            self.backbone.text_model.eval()
        return self

    def pool_geographic_views(self, features: torch.Tensor) -> torch.Tensor:
        """Aggregate [batch, views, dim] for geographic prediction only."""
        if features.ndim != 3 or features.shape[1] < 1:
            raise ValueError("Geographic features must have shape [batch, views, dim]")

        batch_size, num_views, _ = features.shape
        keep = None
        if self.training and num_views > 1 and self.geo_view_dropout > 0:
            keep = torch.rand(batch_size, num_views, device=features.device) >= self.geo_view_dropout
            # Always retain at least one crop per panorama.
            missing = ~keep.any(dim=1)
            if missing.any():
                replacement = torch.randint(num_views, (batch_size,), device=features.device)
                keep[missing, replacement[missing]] = True

        if self.geo_attention is not None:
            scores = self.geo_attention(features).squeeze(-1)
            if keep is not None:
                scores = scores.masked_fill(~keep, float("-inf"))
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            return (features * weights).sum(dim=1)

        if keep is not None:
            return (features * keep.unsqueeze(-1)).sum(dim=1) / keep.sum(dim=1, keepdim=True)
        return features.mean(dim=1)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.backbone.logit_scale

    @property
    def logit_bias(self) -> torch.Tensor:
        return self.backbone.logit_bias

    def encode_images(
        self,
        image_inputs: dict[str, torch.Tensor],
        batch_size: int,
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        view_features = _pooled_features(self.backbone.get_image_features(**image_inputs))
        # Forward values are unchanged. Scale ONLY the geographic gradient
        # entering the vision tower; the geographic head keeps full gradients.
        # Contrastive gradients always use the original view_features.
        detached_features = view_features.detach()
        if self.geo_backbone_gradient_scale == 0:
            # No autograd edge at all: geographic-only runs must not cause
            # AdamW to weight-decay a supposedly detached backbone.
            geo_features = detached_features
        else:
            geo_features = detached_features + self.geo_backbone_gradient_scale * (
                view_features - detached_features
            )
        geo_views = geo_features.reshape(batch_size, num_views, -1)
        panorama_features = self.pool_geographic_views(geo_views)
        geo_logits = self.geo_head(panorama_features)
        image_embeddings = F.normalize(view_features, dim=-1)

        return geo_logits, image_embeddings

    def encode_text(self, text_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        text_features = _pooled_features(self.backbone.get_text_features(**text_inputs))
        return F.normalize(text_features, dim=-1)

    def forward(
        self,
        image_inputs: dict[str, torch.Tensor],
        text_inputs: dict[str, torch.Tensor],
        batch_size: int,
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        geo_logits, image_embeddings = self.encode_images(image_inputs, batch_size, num_views)
        text_embeddings = self.encode_text(text_inputs)
        return geo_logits, image_embeddings, text_embeddings
