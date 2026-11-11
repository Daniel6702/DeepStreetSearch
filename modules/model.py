from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from adaptive_geogrid import AdaptiveGeoGrid
from config import NUM_VIEWS, SIGLIP_MODEL


def _pooled_features(output) -> torch.Tensor:
    """Support Transformers versions returning either a tensor or model output."""
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    raise TypeError(f"Unsupported SigLIP feature output type: {type(output)!r}")


def model_architecture_kwargs(config: dict) -> dict:
    """Rebuild optional heads exactly as recorded in a training checkpoint."""
    return {
        "geo_head": config.get("geo_head", "linear"),
        "geo_pooling": config.get("geo_pooling", "mean"),
        "geo_view_dropout": config.get("geo_view_dropout", 0.0),
        "retrieval_fusion": config.get("retrieval_fusion", "none"),
        "fusion_layers": config.get("fusion_layers", 2),
        "fusion_heads": config.get("fusion_heads", 8),
        "semantic_token_count": config.get("semantic_token_count", 3),
        "context_gate_init": config.get("context_gate_init", 0.05),
    }


class CrossViewTransformer(nn.Module):
    """Contextualize six crop vectors while retaining their local identity.

    The attention bias depends on *circular relative distance*, not absolute
    crop position: rotating the panorama's start view rotates the output too.
    """

    def __init__(self, dim: int, layers: int, heads: int, gate_init: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.input_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, dim)
        self.gate_logit = nn.Parameter(torch.tensor(math.log(gate_init / (1 - gate_init))))
        self.relative_bias = nn.Parameter(torch.zeros(NUM_VIEWS // 2 + 1))
        positions = torch.arange(NUM_VIEWS)
        distances = (positions[:, None] - positions[None, :]).abs()
        self.register_buffer("circular_distances", torch.minimum(distances, NUM_VIEWS - distances),
                             persistent=False)

    def forward(self, crop_vectors: torch.Tensor) -> torch.Tensor:
        if crop_vectors.ndim != 3 or crop_vectors.shape[1] != NUM_VIEWS:
            raise ValueError(f"Cross-view fusion expects [batch, {NUM_VIEWS}, dim]")
        attention_bias = self.relative_bias[self.circular_distances].to(crop_vectors.dtype)
        # The optimized PyTorch TransformerEncoder evaluation fastpath can
        # produce NaNs for this learned floating-point circular attention mask.
        # The ordinary attention path is finite (verified against this checkpoint).
        # Restore the process setting immediately; do not change training behavior.
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        if not self.training and fastpath_enabled:
            torch.backends.mha.set_fastpath_enabled(False)
        try:
            context = self.encoder(self.input_norm(crop_vectors), mask=attention_bias)
        finally:
            if not self.training and fastpath_enabled:
                torch.backends.mha.set_fastpath_enabled(True)
        delta = self.output_projection(self.output_norm(context))
        result = F.normalize(crop_vectors + self.gate_logit.sigmoid() * delta, dim=-1)
        if not self.training and not torch.isfinite(result).all():
            raise RuntimeError("Cross-view inference produced NaN/Inf embeddings")
        return result


class SemanticPanoramaTokens(nn.Module):
    """Learn several independently queryable summaries of all six views."""

    def __init__(self, dim: int, token_count: int, heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(token_count, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.input_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim),
        )

    def forward(self, crop_vectors: torch.Tensor) -> torch.Tensor:
        if crop_vectors.ndim != 3 or crop_vectors.shape[1] != NUM_VIEWS:
            raise ValueError(f"Semantic tokens expect [batch, {NUM_VIEWS}, dim]")
        batch_size = crop_vectors.shape[0]
        # Initialize tokens near the panorama mean in the pretrained SigLIP
        # space, while distinct learned queries allow them to specialize.
        queries = (crop_vectors.mean(dim=1, keepdim=True)
                   + self.queries.unsqueeze(0).expand(batch_size, -1, -1))
        values = self.input_norm(crop_vectors)
        attended, _ = self.attention(queries, values, values, need_weights=False)
        tokens = queries + attended
        return F.normalize(tokens + self.feed_forward(self.output_norm(tokens)), dim=-1)


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
        retrieval_fusion: str = "none",
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        semantic_token_count: int = 3,
        context_gate_init: float = 0.05,
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
        if retrieval_fusion not in {"none", "cross_view", "semantic_tokens"}:
            raise ValueError("retrieval_fusion must be none, cross_view, or semantic_tokens")
        if fusion_layers < 1 or fusion_heads < 1 or semantic_token_count < 1:
            raise ValueError("Fusion layers, heads and token count must be positive")
        if not 0 < context_gate_init < 1:
            raise ValueError("context_gate_init must be in (0, 1)")

        self.geo_view_dropout = geo_view_dropout
        self.geo_pooling = geo_pooling
        self.freeze_text_encoder = freeze_text_encoder
        self.geo_backbone_gradient_scale = geo_backbone_gradient_scale
        self.retrieval_fusion = retrieval_fusion

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
        if vision_dim % fusion_heads and retrieval_fusion != "none":
            raise ValueError(f"Vision dimension {vision_dim} must be divisible by fusion_heads")
        self.cross_view_fusion = (
            CrossViewTransformer(vision_dim, fusion_layers, fusion_heads, context_gate_init)
            if retrieval_fusion == "cross_view" else None
        )
        self.semantic_fusion = (
            SemanticPanoramaTokens(vision_dim, semantic_token_count, fusion_heads)
            if retrieval_fusion == "semantic_tokens" else None
        )
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

    def _encode_all_images(
        self,
        image_inputs: dict[str, torch.Tensor],
        batch_size: int,
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        view_features = _pooled_features(self.backbone.get_image_features(**image_inputs))
        if view_features.shape[0] != batch_size * num_views:
            raise ValueError("Image features do not match batch_size * num_views")
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
        original_views = F.normalize(view_features.reshape(batch_size, num_views, -1), dim=-1)
        contextual_views = original_views
        semantic_tokens = None
        if num_views == NUM_VIEWS:
            if self.cross_view_fusion is not None:
                contextual_views = self.cross_view_fusion(original_views)
            if self.semantic_fusion is not None:
                semantic_tokens = self.semantic_fusion(original_views)

        return (geo_logits, contextual_views.flatten(0, 1), semantic_tokens,
                original_views)

    def encode_images(
        self, image_inputs: dict[str, torch.Tensor], batch_size: int, num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the legacy [batch * crop_views, dim] interface for geo callers."""
        geo_logits, crop_vectors, _, _ = self._encode_all_images(image_inputs, batch_size, num_views)
        return geo_logits, crop_vectors

    def encode_search_images(
        self, image_inputs: dict[str, torch.Tensor], batch_size: int, num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return indexable [batch, retrieval_vectors, dim] and optional raw views."""
        geo_logits, crops, tokens, originals = self._encode_all_images(
            image_inputs, batch_size, num_views,
        )
        retrieval_vectors = crops.reshape(batch_size, num_views, -1)
        if tokens is not None:
            retrieval_vectors = torch.cat((retrieval_vectors, tokens), dim=1)
        raw_views = originals if self.cross_view_fusion is not None and num_views == NUM_VIEWS else None
        return geo_logits, retrieval_vectors, raw_views

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

    def forward_for_training(
        self, image_inputs: dict[str, torch.Tensor], text_inputs: dict[str, torch.Tensor],
        batch_size: int, num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        geo_logits, crops, tokens, _ = self._encode_all_images(image_inputs, batch_size, num_views)
        return geo_logits, crops, self.encode_text(text_inputs), tokens
