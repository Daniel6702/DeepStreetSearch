from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from adaptive_geogrid import AdaptiveGeoGrid
from modules.dataset.collators import TrainingBatchCollator, collate_batch, collate_panorama_crops
from modules.dataset.metadata import load_metadata, precompute_hierarchy_labels
from modules.dataset.shards import SOURCE_SHARD, find_shards, image_key, iter_shard_images



class PanoramaDataset(IterableDataset):
    def __init__(
        self,
        dataset_root: str | Path,
        shuffle: bool = True,
        shuffle_buffer: int = 2000,
        exclude_columns: list[str] | None = None,
        metadata: dict[str, dict[str, str]] | None = None,
        shards: list[str | Path] | None = None,
        grid: AdaptiveGeoGrid | None = None,
        hierarchy_cache: str | Path | None = None,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.exclude_columns = set(exclude_columns or [])

        self.metadata = metadata if metadata is not None else load_metadata(
            self.dataset_root,
            exclude_columns=self.exclude_columns,
        )

        if grid is not None:
            self.precompute_hierarchy_labels(grid, cache_path=hierarchy_cache)

        self.shards = [str(Path(path)) for path in shards] if shards is not None else find_shards(self.dataset_root)
        if not self.shards:
            raise FileNotFoundError(f"No shard_*.tar or shard_*.zip files found in {self.dataset_root / 'images'}")

        self._iteration = 0

    find_shards = staticmethod(find_shards)
    load_metadata = staticmethod(load_metadata)

    def precompute_hierarchy_labels(
        self,
        grid: AdaptiveGeoGrid,
        chunk_size: int = 100_000,
        cache_path: str | Path | None = None,
    ) -> None:
        precompute_hierarchy_labels(self.metadata, grid, chunk_size, cache_path)

    def _make_sample(self, filename: str, image: Image.Image, shard_path: str):
        key = image_key(filename)
        metadata = self.metadata.get(key)

        if metadata is None:
            image.close()
            raise KeyError(f"No metadata found for image: {key}")

        sample_metadata = metadata.copy()
        sample_metadata[SOURCE_SHARD] = shard_path
        return image, sample_metadata

    def _iter_shard(self, shard_path: str, rng: random.Random | None = None) -> Iterator:
        shuffle_buffer = self.shuffle_buffer if rng is not None else 1
        for filename, image in iter_shard_images(shard_path, shuffle_buffer=shuffle_buffer, rng=rng):
            yield self._make_sample(filename, image, shard_path)

    def __iter__(self):
        worker = get_worker_info()
        iteration = self._iteration
        self._iteration += 1

        if worker is None:
            worker_id = 0
            num_workers = 1
            base_seed = torch.initial_seed()
        else:
            worker_id = worker.id
            num_workers = worker.num_workers
            base_seed = worker.seed - worker.id

        base_seed += iteration * 1_000_003
        shards = list(self.shards)

        if self.shuffle:
            random.Random(base_seed).shuffle(shards)

        shards = shards[worker_id::num_workers]

        rng = random.Random(base_seed + worker_id + 1) if self.shuffle else None
        for shard_path in shards:
            yield from self._iter_shard(shard_path, rng=rng)

    def loader(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        prepare_panorama_crops: bool = False,
        model_name: str | None = None,
        descriptions: str | Path | None = None,
        prompt_indices: list[int] | None = None,
    ):
        effective_workers = min(num_workers, len(self.shards)) if num_workers > 0 else 0

        if model_name is not None or descriptions is not None:
            if model_name is None or descriptions is None:
                raise ValueError("model_name and descriptions must be provided together for training")
            if prepare_panorama_crops:
                raise ValueError("prepare_panorama_crops cannot be combined with the training collator")

            collate_fn = TrainingBatchCollator(
                model_name=model_name,
                descriptions=descriptions,
                prompt_indices=prompt_indices,
            )
        elif prepare_panorama_crops:
            collate_fn = collate_panorama_crops
        else:
            collate_fn = collate_batch

        kwargs = {
            "dataset": self,
            "batch_size": batch_size,
            "num_workers": effective_workers,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
        }

        if effective_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = prefetch_factor

        return DataLoader(**kwargs)
