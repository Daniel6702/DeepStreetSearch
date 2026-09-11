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
        shards: list[str | Path] | None = None,
        shuffle: bool = True,
        shuffle_buffer: int = 2000,
        exclude_columns: list[str] | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        exclude_columns = set(exclude_columns or [])

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------

        metadata_path = self.dataset_root / "metadata.csv"

        with metadata_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            self.metadata = {}

            for row in reader:
                # Needed to associate the CSV row with the WebDataset sample.
                image_path = row["image_path"]
                filename = image_path.split("::", 1)[-1]
                key = Path(filename).stem

                metadata = {
                    column: value
                    for column, value in row.items()
                    if column not in exclude_columns
                }

                self.metadata[key] = metadata

        # ------------------------------------------------------------
        # Shards
        # ------------------------------------------------------------

        if shards is None:
            self.shards = sorted(
                str(path)
                for path in (self.dataset_root / "images").glob("shard_*.tar")
            )
        else:
            self.shards = []

            for shard in shards:
                path = Path(shard)

                # Allow either full paths or paths relative to dataset_root.
                if not path.exists():
                    path = self.dataset_root / path

                self.shards.append(str(path))

        if not self.shards:
            raise FileNotFoundError("No dataset shards found")

        # ------------------------------------------------------------
        # WebDataset
        # ------------------------------------------------------------

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

    def _make_sample(self, sample):
        key = Path(sample["__key__"]).name

        if key not in self.metadata:
            raise KeyError(f"No metadata found for image: {key}")

        return sample["jpg"], self.metadata[key]

    def __iter__(self):
        return iter(self.dataset)

    def loader(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        return wds.WebLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_batch,
        )