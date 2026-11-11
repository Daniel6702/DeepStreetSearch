"""Quick CPU smoke checks for the optional panorama modules (no tests/ folder).

Run: python -m tools.smoke_panorama_fusion
"""

import torch
import torch.nn.functional as F
from torch import nn
from types import SimpleNamespace
from unittest.mock import patch

from modules.inference import GeoSearchInference
from modules.losses import DeepStreetSearchLoss
from modules.model import CrossViewTransformer, GeoSearchModel, SemanticPanoramaTokens


def main() -> None:
    torch.manual_seed(11)
    features = F.normalize(torch.randn(2, 6, 32, requires_grad=True), dim=-1)
    texts = F.normalize(torch.randn(12, 32), dim=-1)
    positive_crops = torch.arange(12)

    cross_view = CrossViewTransformer(dim=32, layers=2, heads=4, gate_init=0.05)
    context = cross_view(features)
    assert context.shape == (2, 6, 32)
    assert torch.isfinite(context).all()
    # Circular equivariance: rolling the views must roll the output.
    shifted = cross_view(features.roll(2, dims=1))
    torch.testing.assert_close(shifted, context.roll(2, dims=1), atol=2e-5, rtol=2e-5)
    (context * texts.reshape(2, 6, 32)).sum().backward(retain_graph=True)
    assert cross_view.gate_logit.grad is not None
    assert any(p.grad is not None for p in cross_view.encoder.parameters())

    # The train/eval discrepancy in optimized PyTorch Transformer attention
    # previously created an all-NaN 20k cross-view index. Exercise both paths.
    cross_view.eval()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    with torch.inference_mode():
        eval_context = cross_view(features.detach())
        assert torch.isfinite(eval_context).all()
        torch.testing.assert_close(eval_context, context.detach(), atol=2e-5, rtol=2e-5)
        assert torch.backends.mha.get_fastpath_enabled() == fastpath
    cross_view.train()

    semantic = SemanticPanoramaTokens(dim=32, token_count=3, heads=4)
    tokens = semantic(features)
    assert tokens.shape == (2, 3, 32)
    token_loss = DeepStreetSearchLoss._multi_vector_panorama_loss(
        tokens, texts, positive_crops, torch.tensor(1.0), torch.tensor(-1.0),
    )
    assert torch.isfinite(token_loss)
    token_loss.backward()
    assert semantic.queries.grad is not None
    assert any(p.grad is not None for p in semantic.attention.parameters())

    assert SemanticPanoramaTokens(dim=32, token_count=6, heads=4)(features).shape == (2, 6, 32)

    # Alpha endpoints must exactly reproduce crop-only and token-only ranking.
    searcher = GeoSearchInference.__new__(GeoSearchInference)
    searcher.semantic_view_mode = "all"
    gallery = F.normalize(torch.randn(5, 9, 32), dim=-1)
    captions = F.normalize(torch.randn(2, 32), dim=-1)
    for aggregation in ("mean", "max"):
        searcher.aggregation = aggregation
        searcher.semantic_token_mix = 0.0
        torch.testing.assert_close(searcher._text_retrieval_scores(captions, gallery),
                                   searcher._text_similarity(captions, gallery[:, :6]))
        searcher.semantic_token_mix = 1.0
        torch.testing.assert_close(searcher._text_retrieval_scores(captions, gallery),
                                   searcher._text_similarity(captions, gallery[:, 6:]))
        searcher.semantic_token_mix = 0.3
        assert torch.isfinite(searcher._text_retrieval_scores(captions, gallery)).all()

    # Exercise the actual checkpoint architecture / training / index interfaces
    # without downloading SigLIP weights or depending on the image dataset.
    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=32))
            self.image_model = nn.Linear(32, 32)
            self.text_model = nn.Linear(32, 32)
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
            self.logit_bias = nn.Parameter(torch.tensor(-1.0))

        def get_image_features(self, pixel_values):
            return self.image_model(pixel_values)

        def get_text_features(self, input_ids):
            return self.text_model(input_ids)

    class FakeHierarchy:
        def to_numpy(self, dtype):
            import numpy as np
            return np.arange(4, dtype=dtype).reshape(4, 1)

    class FakeGrid:
        n_classes = (4,)
        n_levels = 1

        def hierarchy_table(self):
            return FakeHierarchy()

    for arch in ("cross_view", "semantic_tokens"):
        with patch("modules.model.AutoModel.from_pretrained", return_value=FakeBackbone()):
            model = GeoSearchModel(
                FakeGrid(), model_name="dummy", geo_head="mlp", geo_pooling="mean",
                freeze_text_encoder=True, retrieval_fusion=arch,
                fusion_heads=4, fusion_layers=2, semantic_token_count=3,
            )
        image_inputs = {"pixel_values": torch.randn(12, 32)}
        text_inputs = {"input_ids": torch.randn(12, 32)}
        geo, crops, captions, extra = model.forward_for_training(
            image_inputs, text_inputs, batch_size=2, num_views=6,
        )
        assert geo.shape == (2, 4) and crops.shape == (12, 32)
        assert captions.shape == (12, 32)
        assert (extra is None) == (arch == "cross_view")
        _, indexed, original = model.encode_search_images(image_inputs, batch_size=2, num_views=6)
        assert indexed.shape == (2, 6 if arch == "cross_view" else 9, 32)
        assert (original is not None) == (arch == "cross_view")
        objective = DeepStreetSearchLoss(
            FakeGrid(), contrastive_weight=2.0, revised_contrastive_positives=True,
            semantic_token_loss_weight=1.0 if arch == "semantic_tokens" else 0.0,
        )
        loss, parts = objective(
            geo, torch.tensor([[0], [1]]), crops, captions, positive_crops,
            model.logit_scale, model.logit_bias, semantic_tokens=extra,
        )
        assert torch.isfinite(loss) and torch.isfinite(parts["semantic_contrastive"])
        loss.backward()
        assert model.backbone.image_model.weight.grad is not None
        # Strict state loading is required by every downstream evaluation stage.
        model.load_state_dict(model.state_dict(), strict=True)
    print("Panorama architecture smoke checks passed")


if __name__ == "__main__":
    main()
