from __future__ import annotations

import csv
import random
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

def _collate_batch(batch):
    """Keep images as PIL images and metadata as dictionaries."""
    images, metadata = zip(*batch)
    return list(images), list(metadata)


def _decode_image(fileobj: BinaryIO) -> Image.Image:
    """Fully decode an image and detach it from the archive stream."""
    with Image.open(fileobj) as image:
        image.load()

        if image.mode != "RGB":
            return image.convert("RGB")

        return image.copy()


def _image_key(filename: str) -> str:
    """
    Convert an image filename/path to the metadata lookup key.

    Examples:
        abc123.jpg       -> abc123
        abc123_360.jpg   -> abc123
    """
    key = Path(filename).stem

    if key.endswith("_360"):
        key = key[:-4]

    return key


def _buffer_shuffle(iterator: Iterator, buffer_size: int, rng: random.Random):
    """Streaming shuffle without loading the entire dataset into memory."""
    if buffer_size <= 1:
        yield from iterator
        return

    buffer = []

    for sample in iterator:
        if len(buffer) < buffer_size:
            buffer.append(sample)
            continue

        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = sample

    rng.shuffle(buffer)
    yield from buffer


def _close_prepared_crops(crops_batch) -> None:
    for crops in crops_batch:
        for crop in crops:
            crop.image.close()


class PanoramaDataset(IterableDataset):
    def __init__(
        self,
        dataset_root: str | Path,
        shuffle: bool = True,
        shuffle_buffer: int = 2000,
        exclude_columns: list[str] | None = None,
        metadata: dict[str, dict[str, str]] | None = None,
        shards: list[str | Path] | None = None,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.exclude_columns = set(exclude_columns or [])

        self.metadata = (
            metadata
            if metadata is not None
            else self.load_metadata(
                self.dataset_root,
                exclude_columns=self.exclude_columns,
            )
        )

        self.shards = (
            [str(Path(path)) for path in shards]
            if shards is not None
            else self.find_shards(self.dataset_root)
        )

        if not self.shards:
            raise FileNotFoundError(
                f"No shard_*.tar or shard_*.zip files found in "
                f"{self.dataset_root / 'images'}"
            )

        self._iteration = 0

    @staticmethod
    def find_shards(dataset_root: str | Path) -> list[str]:
        dataset_root = Path(dataset_root)
        images_dir = dataset_root / "images"

        shards = [
            *images_dir.glob("shard_*.tar"),
            *images_dir.glob("shard_*.zip"),
        ]

        return sorted(str(path) for path in shards)

    @staticmethod
    def load_metadata(
        dataset_root: str | Path,
        exclude_columns: set[str] | list[str] | None = None,
    ) -> dict[str, dict[str, str]]:
        dataset_root = Path(dataset_root)
        exclude_columns = set(exclude_columns or [])
        metadata_path = dataset_root / "metadata.csv"

        with metadata_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as f:
            reader = csv.DictReader(f)
            metadata: dict[str, dict[str, str]] = {}

            for row in reader:
                image_path = row.get("image_path")

                if image_path:
                    filename = image_path.split("::", 1)[-1]
                    key = _image_key(filename)
                else:
                    panoid = row.get("panoid")

                    if not panoid:
                        raise ValueError(
                            "metadata.csv row contains neither "
                            "'image_path' nor 'panoid', so it cannot be "
                            "matched to an image."
                        )

                    key = panoid

                metadata[key] = {
                    column: value
                    for column, value in row.items()
                    if column not in exclude_columns
                }

        return metadata

    def _make_sample(self, filename: str, image: Image.Image):
        key = _image_key(filename)
        metadata = self.metadata.get(key)

        if metadata is None:
            image.close()
            raise KeyError(f"No metadata found for image: {key}")

        return image, metadata

    def _iter_tar(self, shard_path: str) -> Iterator:
        with tarfile.open(shard_path, mode="r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue

                if Path(member.name).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                fileobj = archive.extractfile(member)

                if fileobj is None:
                    continue

                with fileobj:
                    image = _decode_image(fileobj)

                yield self._make_sample(member.name, image)

    def _iter_zip(self, shard_path: str) -> Iterator:
        with zipfile.ZipFile(shard_path, mode="r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue

                if Path(member.filename).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                with archive.open(member, mode="r") as fileobj:
                    image = _decode_image(fileobj)

                yield self._make_sample(member.filename, image)

    def _iter_shard(self, shard_path: str) -> Iterator:
        suffix = Path(shard_path).suffix.lower()

        if suffix == ".tar":
            yield from self._iter_tar(shard_path)
        elif suffix == ".zip":
            yield from self._iter_zip(shard_path)
        else:
            raise ValueError(f"Unsupported shard format: {shard_path}")

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

        # Each DataLoader worker receives distinct shards.
        shards = shards[worker_id::num_workers]

        def samples():
            for shard_path in shards:
                yield from self._iter_shard(shard_path)

        iterator = samples()

        if self.shuffle:
            iterator = _buffer_shuffle(
                iterator,
                self.shuffle_buffer,
                random.Random(base_seed + worker_id + 1),
            )

        yield from iterator

    def loader(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        prepare_panorama_crops: bool = False,
    ):
        # A worker processes whole shards. More workers than shards provide no
        # benefit and merely create idle processes.
        effective_workers = (
            min(num_workers, len(self.shards))
            if num_workers > 0
            else 0
        )

        if prepare_panorama_crops:
            from modules.panorama import split_panorama

            def collate_fn(batch):
                images, metadata = zip(*batch)
                crops_batch = []

                try:
                    for image in images:
                        try:
                            crops_batch.append(split_panorama(image))
                        finally:
                            image.close()
                except Exception:
                    _close_prepared_crops(crops_batch)
                    raise

                return crops_batch, list(metadata)
        else:
            collate_fn = _collate_batch

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
