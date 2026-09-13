from __future__ import annotations

from pathlib import Path
import csv

import webdataset as wds


def _collate_batch(batch):
    """Keep images as PIL images and metadata as dictionaries."""
    images, metadata = zip(*batch)
    return list(images), list(metadata)


class PanoramaDataset:
    def __init__(
        self,
        dataset_root: str | Path,
        shuffle: bool = True,
        shuffle_buffer: int = 2000,
        exclude_columns: list[str] | None = None,
        metadata: dict[str, dict[str, str]] | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.exclude_columns = set(exclude_columns or [])

        self.metadata = metadata or self.load_metadata(
            self.dataset_root,
            exclude_columns=self.exclude_columns,
        )

        self.shards = self.find_shards(self.dataset_root)

        if not self.shards:
            raise FileNotFoundError("No dataset shards found")
        
        dataset = wds.WebDataset(
            self.shards,
            shardshuffle=shuffle,
        )
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer)
        self.dataset = (
            dataset
            .decode("pil")
            .map(self._make_sample)
        )

    @staticmethod
    def find_shards(dataset_root: str | Path) -> list[str]:
        dataset_root = Path(dataset_root)
        return sorted(
            str(path)
            for path in (dataset_root / "images").glob("shard_*.tar")
        )

    @staticmethod
    def load_metadata(
        dataset_root: str | Path,
        exclude_columns: set[str] | list[str] | None = None,
    ) -> dict[str, dict[str, str]]:
        dataset_root = Path(dataset_root)
        exclude_columns = set(exclude_columns or [])
        metadata_path = dataset_root / "metadata.csv"

        with metadata_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            metadata: dict[str, dict[str, str]] = {}

            for row in reader:
                image_path = row["image_path"]
                filename = image_path.split("::", 1)[-1]
                key = Path(filename).stem

                metadata[key] = {
                    column: value
                    for column, value in row.items()
                    if column not in exclude_columns
                }

        return metadata

    def _make_sample(self, sample):
        key = Path(sample["__key__"]).name

        if key not in self.metadata:
            raise KeyError(f"No metadata found for image: {key}")

        return sample["jpg"], self.metadata[key]

    def __iter__(self):
        return iter(self.dataset)

    def loader(self, batch_size: int = 1, num_workers: int = 4):
        return wds.WebLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_batch,
        )