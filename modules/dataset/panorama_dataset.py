from __future__ import annotations

import math
import random
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from adaptive_geogrid import AdaptiveGeoGrid
from modules.dataset.collators import TrainingBatchCollator, collate_batch, collate_panorama_crops
from modules.dataset.metadata import load_metadata, precompute_hierarchy_labels
from modules.dataset.shards import IMAGE_EXTENSIONS, SOURCE_SHARD, find_shards, image_key, iter_shard_images



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
        fraction: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.exclude_columns = set(exclude_columns or [])
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction
        self.seed = seed

        self.metadata = metadata if metadata is not None else load_metadata(
            self.dataset_root,
            exclude_columns=self.exclude_columns,
        )

        if grid is not None:
            self.precompute_hierarchy_labels(grid, cache_path=hierarchy_cache)

        self.shards = [str(Path(path)) for path in shards] if shards is not None else find_shards(self.dataset_root)
        if not self.shards:
            raise FileNotFoundError(f"No shard_*.tar or shard_*.zip files found in {self.dataset_root / 'images'}")

        # Choose once. Subsequent epochs reshuffle the same panorama IDs.
        if not self.metadata:
            raise ValueError("Training metadata is empty")
        sample_count = max(1, int(len(self.metadata) * fraction))
        self.selected_keys = (
            set(random.Random(seed).sample(list(self.metadata), sample_count))
            if fraction < 1.0 else set(self.metadata)
        )
        self.num_samples = sample_count
        self._shard_counts = None
        self._iteration = 0
        # Shared so set_epoch() reaches persistent DataLoader workers.
        self._requested_epoch = torch.full((1,), -1, dtype=torch.long).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._requested_epoch[0] = epoch

    def _epoch_shards(self, epoch: int) -> list[str]:
        shards = list(self.shards)
        if self.shuffle:
            random.Random((self.seed or 0) + epoch * 1_000_003).shuffle(shards)
        return shards

    def _get_shard_counts(self) -> dict[str, int]:
        if self._shard_counts is not None:
            return self._shard_counts

        counts = {path: 0 for path in self.shards}
        paths_by_name = {Path(path).name: path for path in self.shards}
        selected = self.selected_keys
        missing_paths = False
        for key, row in self.metadata.items():
            if selected is not None and key not in selected:
                continue
            image_path = row.get("image_path")
            if not image_path:
                missing_paths = True
                break
            filename = Path(image_path.split("::", 1)[0]).name
            if filename not in paths_by_name:
                raise ValueError(f"Metadata references unknown image shard: {image_path}")
            counts[paths_by_name[filename]] += 1

        if missing_paths:
            # Older metadata may contain only panoids. Inspect archive headers,
            # not image bytes, to count actual samples for the scheduler.
            counts = {}
            for path in self.shards:
                if Path(path).suffix.lower() == ".tar":
                    with tarfile.open(path, "r:*") as archive:
                        names = (m.name for m in archive if m.isfile())
                        counts[path] = sum(
                            Path(name).suffix.lower() in IMAGE_EXTENSIONS
                            and (selected is None or image_key(name) in selected)
                            for name in names
                        )
                else:
                    with zipfile.ZipFile(path) as archive:
                        counts[path] = sum(
                            not m.is_dir() and Path(m.filename).suffix.lower() in IMAGE_EXTENSIONS
                            and (selected is None or image_key(m.filename) in selected)
                            for m in archive.infolist()
                        )

        if sum(counts.values()) != self.num_samples:
            raise ValueError(
                f"Expected {self.num_samples} panoramas but shard counts total {sum(counts.values())}. "
                "Check metadata.csv image paths and available shards."
            )
        self._shard_counts = counts
        return counts

    def steps_per_epoch(self, batch_size: int, num_workers: int, epoch: int) -> int:
        """Exact number of batches for a shard-partitioned IterableDataset."""
        if batch_size <= 0 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.shuffle and self.seed is None:
            raise ValueError("steps_per_epoch requires a fixed seed when shuffle is enabled")
        workers = min(num_workers, len(self.shards)) if num_workers > 0 else 1
        shards = self._epoch_shards(epoch)
        counts = self._get_shard_counts()
        return sum(
            math.ceil(sum(counts[path] for path in shards[worker::workers]) / batch_size)
            for worker in range(workers)
        )

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
        for filename, image in iter_shard_images(
            shard_path, shuffle_buffer=shuffle_buffer, rng=rng, allowed_keys=self.selected_keys
        ):
            yield self._make_sample(filename, image, shard_path)

    def __iter__(self):
        worker = get_worker_info()
        requested_epoch = int(self._requested_epoch[0])
        if requested_epoch >= 0:
            iteration = requested_epoch
        else:
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

        if self.seed is not None:
            base_seed = self.seed
        base_seed += iteration * 1_000_003
        shards = self._epoch_shards(iteration) if self.seed is not None else list(self.shards)
        if self.shuffle and self.seed is None:
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
        description_mode: str = "all",
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
                description_mode=description_mode,
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
