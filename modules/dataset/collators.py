from __future__ import annotations

import random
from pathlib import Path

import torch
from config import NUM_VIEWS
from modules.dataset.metadata import HIERARCHY_LABELS
from modules.dataset.shards import SOURCE_SHARD
from modules.descriptions import DescriptionStore
from modules.panorama import split_panorama


def collate_batch(batch):
    """Keep images as PIL images and metadata as dictionaries."""
    images, metadata = zip(*batch)
    return list(images), list(metadata)


def close_prepared_crops(crops_batch) -> None:
    for crops in crops_batch:
        for crop in crops:
            crop.image.close()


def collate_panorama_crops(batch):
    images, metadata = zip(*batch)
    crops_batch = []

    try:
        for image in images:
            try:
                crops_batch.append(split_panorama(image))
            finally:
                image.close()
    except Exception:
        close_prepared_crops(crops_batch)
        raise

    return crops_batch, list(metadata)


class TrainingBatchCollator:
    """Prepare panorama views, crop-paired descriptions, and hierarchy labels."""

    def __init__(
        self,
        model_name: str,
        descriptions: str | Path,
        prompt_indices: list[int] | None = None,
    ):
        self.model_name = model_name
        self.descriptions = Path(descriptions)
        self.prompt_indices = prompt_indices
        self.processor = None
        self.stores: dict[Path, DescriptionStore] = {}
        self.rng = None

    def _get_processor(self):
        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_name)
        return self.processor

    def _get_rng(self):
        if self.rng is None:
            self.rng = random.Random(torch.initial_seed())
        return self.rng

    def _database_path(self, metadata: dict[str, str]) -> Path:
        if self.descriptions.is_file():
            return self.descriptions

        shard_name = Path(metadata[SOURCE_SHARD]).name
        return self.descriptions / f"{shard_name}.sqlite"

    def _get_store(self, path: Path) -> DescriptionStore:
        store = self.stores.get(path)
        if store is None:
            if not path.exists():
                raise FileNotFoundError(f"Description database not found: {path}")
            store = DescriptionStore(path, read_only=True)
            self.stores[path] = store
        return store

    def _sample_descriptions(self, metadata: list[dict[str, str]], crops_batch) -> list[str]:
        by_database: dict[Path, list[tuple[str, int]]] = {}
        ordered_keys: list[tuple[str, int]] = []

        for row, crops in zip(metadata, crops_batch):
            panoid = row.get("panoid")
            if not panoid:
                raise KeyError("Training metadata must contain 'panoid'")

            database_path = self._database_path(row)
            keys = [(panoid, crop.subimage_index) for crop in crops]
            by_database.setdefault(database_path, []).extend(keys)
            ordered_keys.extend(keys)

        sampled: dict[tuple[str, int], str] = {}
        rng = self._get_rng()

        for database_path, crop_keys in by_database.items():
            store = self._get_store(database_path)
            sampled.update(store.sample_many(crop_keys, rng, self.prompt_indices))

        return [sampled[key] for key in ordered_keys]

    def __call__(self, batch):
        panoramas, metadata = zip(*batch)
        metadata = list(metadata)
        crops_batch = []
        views = []

        try:
            for panorama in panoramas:
                try:
                    crops = split_panorama(panorama)
                    crops_batch.append(crops)
                    views.extend(crop.image for crop in crops)
                finally:
                    panorama.close()

            descriptions = self._sample_descriptions(metadata, crops_batch)
            processor = self._get_processor()

            processed_images = processor(images=views, return_tensors="pt")
            processed_text = processor(text=descriptions, padding="max_length", truncation=True, return_tensors="pt")

            image_inputs = {key: value for key, value in processed_images.items() if torch.is_tensor(value)}
            text_inputs = {key: value for key, value in processed_text.items() if torch.is_tensor(value)}

            try:
                labels = torch.tensor([row[HIERARCHY_LABELS] for row in metadata], dtype=torch.long)
            except KeyError as exc:
                raise RuntimeError(
                    "Hierarchy labels have not been precomputed. "
                    "Construct PanoramaDataset with grid=... before creating the training loader."
                ) from exc

            return image_inputs, text_inputs, labels, metadata, NUM_VIEWS
        finally:
            close_prepared_crops(crops_batch)
