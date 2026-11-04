"""Reusable text/image-to-location inference against a saved panorama index.

The index is built separately (currently by text_location_eval.indexing). Nothing
in this module depends on evaluation queries, judges, or ground-truth metrics.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path


def load_reference_index(path: str | Path) -> dict:
    """Load and validate the persisted [panoramas, views, features] gallery."""
    import torch

    index = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(index, dict) or not {"signature", "records", "embeddings"} <= index.keys():
        raise ValueError(f"Invalid reference index: {path}")
    embeddings = index["embeddings"]
    if embeddings.ndim != 3 or embeddings.shape[1] != 6:
        raise ValueError("Index embeddings must have shape [panoramas, 6, dimensions]")
    if len(index["records"]) != embeddings.shape[0]:
        raise ValueError("Reference index records and embeddings differ in length")
    return index


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_gallery_indices(total: int, size: int | None, seed: int = 42,
                           image_ids: list[str] | None = None) -> range | list[int]:
    """Nested random gallery subsets: sizes A < B share the same first A draws.

    Sort selected positions for efficient sequential reads of the saved index.
    A new size never changes which panoramas belong to a smaller subset.
    If image_ids are given, selection is also independent of index row order,
    allowing different checkpoints to use exactly the same geographic gallery.
    """
    if total < 1 or seed < 0:
        raise ValueError("Gallery must be nonempty and seed nonnegative")
    if size is None:
        size = total
    if not 1 <= size <= total:
        raise ValueError(f"Gallery size must be between 1 and {total:,}; got {size}")
    if image_ids is not None and (len(image_ids) != total or len(set(image_ids)) != total):
        raise ValueError("Gallery image IDs must be unique and match the index length")
    if size == total:
        return range(total)
    shuffled = (sorted(range(total), key=lambda i: image_ids[i])
                if image_ids is not None else list(range(total)))
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:size])


class GeoSearchInference:
    """Load a trained DeepStreetSearch model and retrieve ranked coordinates.

    `gallery_size=None` searches the full index. Smaller sizes use repeatable,
    nested random subsets of the *same* index without recomputing embeddings.
    """

    def __init__(self, checkpoint: str | Path, index: str | Path,
                 grid: str | Path, model_name: str,
                 device: str = "cuda", gallery_size: int | None = None,
                 gallery_seed: int = 42, aggregation: str = "mean",
                 reference_chunk_size: int = 2048, text_batch_size: int = 16,
                 expected_signature: dict | None = None) -> None:
        import torch
        from transformers import AutoProcessor

        from adaptive_geogrid import load_grid
        from modules.model import GeoSearchModel

        if aggregation not in {"mean", "max"}:
            raise ValueError("aggregation must be 'mean' or 'max'")
        if reference_chunk_size < 1 or text_batch_size < 1:
            raise ValueError("Batch and chunk sizes must be positive")

        self.index = load_reference_index(index)
        signature = self.index["signature"]
        if expected_signature is not None and signature != expected_signature:
            raise ValueError("Checkpoint, grid, model, or data differs from the index. Rebuild the index first.")

        # Standalone applications needn't know the dataset/seed used for indexing,
        # but must never silently combine embeddings from incompatible weights.
        if (signature.get("checkpoint_sha256") != _file_sha256(checkpoint)
                or signature.get("grid_sha256") != _file_sha256(grid)
                or signature.get("model") != model_name):
            raise ValueError("Checkpoint, grid, or model differs from the index")

        self.records = self.index["records"]
        self.gallery_indices = select_gallery_indices(
            len(self.records), gallery_size, gallery_seed,
            image_ids=([record["image_id"] for record in self.records]
                       if gallery_size is not None else None))
        self.gallery_count = len(self.gallery_indices)
        self.aggregation = aggregation
        self.reference_chunk_size = reference_chunk_size
        self.text_batch_size = text_batch_size
        self.device = device
        self.torch = torch
        self._grid_path = Path(grid)
        self._geo_centers = None

        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        train_config = checkpoint_data.get("run_config") or {}
        self.model = GeoSearchModel(
            load_grid(grid), model_name=model_name,
            geo_head=train_config.get("geo_head", "linear"),
            geo_pooling=train_config.get("geo_pooling", "mean"),
            geo_view_dropout=train_config.get("geo_view_dropout", 0.0),
        ).to(device).eval()
        self.model.load_state_dict(checkpoint_data["model"])
        del checkpoint_data
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.text_limit = self.model.backbone.config.text_config.max_position_embeddings

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Return the top-K georeferenced matches for one free-form query."""
        return self.retrieve_many([query], top_k=top_k)[0]["matches"]

    def retrieve_many(self, queries: list[str], top_k: int = 10) -> list[dict]:
        """Retrieve batches of queries; include tokenizer truncation metadata."""
        import torch.nn.functional as F

        if top_k < 1 or top_k > self.gallery_count:
            raise ValueError(f"top_k must be between 1 and gallery size {self.gallery_count:,}")
        if not queries:
            return []
        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("Queries must be nonempty strings")

        torch = self.torch
        lengths = [len(tokens) for tokens in self.processor.tokenizer(
            queries, truncation=False, add_special_tokens=True)["input_ids"]]
        if any(length > self.text_limit for length in lengths):
            print(f"Warning: {sum(length > self.text_limit for length in lengths)} queries exceed "
                  f"the model's {self.text_limit}-token limit and will be truncated.")

        with torch.inference_mode():
            parts = []
            for start in range(0, len(queries), self.text_batch_size):
                inputs = self.processor(text=queries[start:start + self.text_batch_size],
                                        padding="max_length", truncation=True,
                                        return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()
                          if key in {"input_ids", "attention_mask"}}
                parts.append(self.model.encode_text(inputs).float())
            text_features = F.normalize(torch.cat(parts, dim=0), dim=-1)

            best_scores = torch.full((len(queries), top_k), -float("inf"), device=self.device)
            best_ids = torch.full((len(queries), top_k), -1, dtype=torch.long,
                                  device=self.device)
            full_gallery = isinstance(self.gallery_indices, range)
            for start in range(0, self.gallery_count, self.reference_chunk_size):
                if full_gallery:
                    ids_cpu = range(start, min(start + self.reference_chunk_size, self.gallery_count))
                    crops = self.index["embeddings"][ids_cpu.start:ids_cpu.stop]
                    ids = torch.arange(ids_cpu.start, ids_cpu.stop, device=self.device)
                else:
                    ids_cpu = self.gallery_indices[start:start + self.reference_chunk_size]
                    crops = self.index["embeddings"].index_select(
                        0, torch.tensor(ids_cpu, dtype=torch.long))
                    ids = torch.tensor(ids_cpu, device=self.device, dtype=torch.long)

                crops = F.normalize(crops.float().to(self.device), dim=-1)
                if self.aggregation == "mean":
                    panorama_features = F.normalize(crops.mean(dim=1), dim=-1)
                    scores = text_features @ panorama_features.T
                else:
                    scores = torch.einsum("qd,nvd->qnv", text_features, crops).amax(dim=-1)

                ids = ids.unsqueeze(0).expand(len(queries), -1)
                merged_scores = torch.cat((best_scores, scores), dim=1)
                merged_ids = torch.cat((best_ids, ids), dim=1)
                best_scores, positions = merged_scores.topk(top_k, dim=1)
                best_ids = merged_ids.gather(1, positions)

        results = []
        for qidx, query in enumerate(queries):
            matches = [
                {"rank": rank, "similarity": float(score), **self.records[item]}
                for rank, (score, item) in enumerate(
                    zip(best_scores[qidx].tolist(), best_ids[qidx].tolist()), 1)
            ]
            results.append({"query": query, "matches": matches,
                            "truncated": lengths[qidx] > self.text_limit,
                            "token_length": lengths[qidx]})
        return results

    def _encode_query_image(self, image, panorama: bool = False):
        """Encode a PIL image or file path as either one crop or six panorama views.

        The geographic head supports one or multiple views; a normal photo is
        never treated as an equirectangular panorama unless explicitly requested.
        """
        from PIL import Image

        from modules.panorama import split_panorama

        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                prepared = source.convert("RGB")
        elif isinstance(image, Image.Image):
            prepared = image.convert("RGB")
        else:
            raise TypeError("image must be a PIL.Image.Image or image file path")

        crops = []
        try:
            if panorama:
                crops = [crop.image for crop in split_panorama(prepared)]
            else:
                crops = [prepared]

            with self.torch.inference_mode():
                inputs = self.processor(images=crops, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()
                          if self.torch.is_tensor(value)}
                logits, embeddings = self.model.encode_images(inputs, 1, len(crops))
            return logits.float(), embeddings.float()
        finally:
            for crop in crops:
                if crop is not prepared:
                    crop.close()
            prepared.close()

    def retrieve_image(self, image, top_k: int = 10, panorama: bool = False) -> list[dict]:
        """Find index panoramas visually similar to a photo or full panorama.

        `image` is a file path or PIL image. Set `panorama=True` only for an
        equirectangular panorama in the dataset's expected dimensions.
        `aggregation='mean'` compares averaged view embeddings; 'max' matches
        the strongest pair of query/index crops.
        """
        import torch.nn.functional as F

        if not 1 <= top_k <= self.gallery_count:
            raise ValueError(f"top_k must be between 1 and gallery size {self.gallery_count:,}")

        _, features = self._encode_query_image(image, panorama=panorama)
        torch = self.torch
        with torch.inference_mode():
            query = F.normalize(features.to(self.device), dim=-1)
            if self.aggregation == "mean":
                query = F.normalize(query.mean(dim=0, keepdim=True), dim=-1)

            best_scores = torch.full((top_k,), -float("inf"), device=self.device)
            best_ids = torch.full((top_k,), -1, dtype=torch.long, device=self.device)
            full_gallery = isinstance(self.gallery_indices, range)
            for start in range(0, self.gallery_count, self.reference_chunk_size):
                if full_gallery:
                    stop = min(start + self.reference_chunk_size, self.gallery_count)
                    crops = self.index["embeddings"][start:stop]
                    ids = torch.arange(start, stop, device=self.device)
                else:
                    selected = self.gallery_indices[start:start + self.reference_chunk_size]
                    crops = self.index["embeddings"].index_select(
                        0, torch.tensor(selected, dtype=torch.long))
                    ids = torch.tensor(selected, dtype=torch.long, device=self.device)

                crops = F.normalize(crops.float().to(self.device), dim=-1)
                if self.aggregation == "mean":
                    gallery = F.normalize(crops.mean(dim=1), dim=-1)
                    scores = (query @ gallery.T).squeeze(0)
                else:
                    scores = torch.einsum("qd,nvd->qnv", query, crops).amax(dim=(0, 2))

                scores, positions = torch.cat((best_scores, scores)).topk(top_k)
                best_ids = torch.cat((best_ids, ids)).gather(0, positions)
                best_scores = scores

        return [
            {"rank": rank, "similarity": float(score), **self.records[row_id]}
            for rank, (score, row_id) in enumerate(zip(best_scores.tolist(), best_ids.tolist()), 1)
        ]

    def predict_image_location(self, image, top_k: int = 5,
                               panorama: bool = False) -> list[dict]:
        """Return the geographic head's most probable grid cells for an image.

        Unlike `retrieve_image`, this does not use the reference index for
        ranking. Coordinates are representative points of the predicted cells.
        """
        from adaptive_geogrid import load_grid

        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self._geo_centers is None:
            grid = load_grid(self._grid_path)
            tiles = grid.level(0).sort_values("tile_id")
            ids = tiles["tile_id"].to_numpy()
            import numpy as np
            if not np.array_equal(ids, np.arange(grid.n_classes[0])):
                raise ValueError("Fine grid tile IDs must be contiguous starting from zero")
            points = tiles.geometry.representative_point()
            self._geo_centers = list(zip(points.y.to_numpy(), points.x.to_numpy()))

        logits, _ = self._encode_query_image(image, panorama=panorama)
        probs = logits.softmax(dim=-1).squeeze(0)
        values, indices = probs.topk(min(top_k, len(self._geo_centers)))
        return [
            {"rank": rank, "cell_id": int(cell_id),
             "probability": float(probability),
             "latitude": float(self._geo_centers[cell_id][0]),
             "longitude": float(self._geo_centers[cell_id][1])}
            for rank, (probability, cell_id) in enumerate(zip(values.tolist(), indices.tolist()), 1)
        ]
