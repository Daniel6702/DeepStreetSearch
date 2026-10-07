from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel, SiglipTextModel

from config import NUM_GEO_CLASSES, NUM_EMBED_CLASSES

class VisualEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-384",
        freeze_backbone: bool = False,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()

        self.backbone = SiglipVisionModel.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
        )

        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # Geographic classification.
        self.geo_head = nn.Linear(hidden_size, NUM_GEO_CLASSES)

        # Shared image/text retrieval space.
        self.embedding_head = nn.Linear(hidden_size, NUM_EMBED_CLASSES)

    def forward(
        self,
        image_inputs: dict[str, torch.Tensor],
        batch_size: int,
        num_views: int,
    ):
        outputs = self.backbone(**image_inputs)

        features = outputs.pooler_output

        # [B * V, H] -> [B, V, H]
        features = features.reshape(batch_size, num_views, -1)

        # Aggregate the different panorama views.
        features = features.mean(dim=1)

        geo_logits = self.geo_head(features)

        embeddings = self.embedding_head(features)
        embeddings = F.normalize(embeddings, dim=-1)

        return geo_logits, embeddings


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-384",
        freeze_backbone: bool = False,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()

        self.backbone = SiglipTextModel.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
        )

        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # Same dimensionality as VisualEncoder.embedding_head.
        self.embedding_head = nn.Linear(hidden_size, NUM_EMBED_CLASSES)

    def forward(self, text_inputs: dict[str, torch.Tensor]):
        outputs = self.backbone(**text_inputs)

        features = outputs.pooler_output

        embeddings = self.embedding_head(features)
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings


class GeoSearchModel(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-384",
        freeze_backbone: bool = False,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()

        self.visual_encoder = VisualEncoder(
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            attn_implementation=attn_implementation,
        )

        self.text_encoder = TextEncoder(
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            attn_implementation=attn_implementation,
        )

    def encode_images(
        self,
        image_inputs: dict[str, torch.Tensor],
        batch_size: int,
        num_views: int,
    ):
        return self.visual_encoder(
            image_inputs,
            batch_size,
            num_views,
        )

    def encode_text(self, text_inputs: dict[str, torch.Tensor]):
        return self.text_encoder(text_inputs)