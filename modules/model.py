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
    """SigLIP image/text encoder with a geographic head on panorama features."""

    def __init__(
        self,
        grid: AdaptiveGeoGrid,
        model_name: str = SIGLIP_MODEL,
        freeze_backbone: bool = False,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()

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

        vision_dim = self.backbone.config.vision_config.hidden_size
        self.geo_head = nn.Linear(vision_dim, grid.n_classes[0])

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
        panorama_features = view_features.reshape(batch_size, num_views, -1).mean(dim=1)

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
