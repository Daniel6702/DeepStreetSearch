from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from adaptive_geogrid import AdaptiveGeoGrid
from modules.dataset.shards import image_key

HIERARCHY_LABELS = "__hierarchy_labels__"


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
            image_path = row.get("image_path")

            if image_path:
                filename = image_path.split("::", 1)[-1]
                key = image_key(filename)
            else:
                panoid = row.get("panoid")
                if not panoid:
                    raise ValueError(
                        "metadata.csv row contains neither 'image_path' nor 'panoid', "
                        "so it cannot be matched to an image."
                    )
                key = panoid

            metadata[key] = {
                column: value
                for column, value in row.items()
                if column not in exclude_columns
            }

    return metadata


def coordinates(rows: list[dict[str, str]]) -> np.ndarray:
    result = np.empty((len(rows), 2), dtype=np.float64)

    for index, row in enumerate(rows):
        lat = row.get("lat", row.get("pano_lat"))
        lon = row.get("lon", row.get("pano_lon"))

        if lat is None or lon is None:
            raise KeyError("Metadata must contain lat/lon or pano_lat/pano_lon")

        result[index] = (float(lon), float(lat))

    return result


def _metadata_fingerprint(metadata: dict[str, dict[str, str]]) -> str:
    digest = hashlib.sha256()

    for key, row in metadata.items():
        lat = row.get("lat", row.get("pano_lat"))
        lon = row.get("lon", row.get("pano_lon"))
        digest.update(f"{key}\0{lat}\0{lon}\n".encode())

    return digest.hexdigest()


def _assign_hierarchy_labels(metadata: dict[str, dict[str, str]], labels: np.ndarray) -> None:
    for row, path in zip(metadata.values(), labels):
        row[HIERARCHY_LABELS] = tuple(int(value) for value in path)


def _load_hierarchy_labels(
    metadata: dict[str, dict[str, str]],
    grid: AdaptiveGeoGrid,
    cache_path: Path,
) -> bool:
    if not cache_path.is_file():
        return False

    with np.load(cache_path, allow_pickle=False) as cache:
        labels = cache["labels"]
        n_classes = tuple(int(value) for value in cache["n_classes"])
        fingerprint = str(cache["metadata_fingerprint"].item())

    expected_shape = (len(metadata), grid.n_levels)
    if labels.shape != expected_shape:
        raise ValueError(f"Geographic label cache has shape {labels.shape}, expected {expected_shape}: {cache_path}")
    if n_classes != grid.n_classes:
        raise ValueError(f"Geographic label cache was created for grid classes {n_classes}, current grid has {grid.n_classes}: {cache_path}")
    if fingerprint != _metadata_fingerprint(metadata):
        raise ValueError(f"Geographic label cache does not match the current metadata: {cache_path}")

    _assign_hierarchy_labels(metadata, labels)
    print(f"Loaded geographic labels: {cache_path}")
    return True


def _save_hierarchy_labels(
    metadata: dict[str, dict[str, str]],
    grid: AdaptiveGeoGrid,
    labels: np.ndarray,
    cache_path: Path,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("wb") as f:
        np.savez(
            f,
            labels=labels,
            n_classes=np.asarray(grid.n_classes, dtype=np.int64),
            metadata_fingerprint=np.asarray(_metadata_fingerprint(metadata)),
        )

    print(f"Saved geographic labels: {cache_path}")


def precompute_hierarchy_labels(
    metadata: dict[str, dict[str, str]],
    grid: AdaptiveGeoGrid,
    chunk_size: int = 100_000,
    cache_path: str | Path | None = None,
) -> None:
    """Load cached geographic labels or classify the metadata once."""
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and _load_hierarchy_labels(metadata, grid, cache_path):
        return

    keys = list(metadata)
    cached_labels = np.empty((len(keys), grid.n_levels), dtype=np.int64) if cache_path is not None else None
    progress = tqdm(total=len(keys), desc="Classifying metadata", unit="samples")

    for start in range(0, len(keys), chunk_size):
        chunk_keys = keys[start:start + chunk_size]
        rows = [metadata[key] for key in chunk_keys]
        labels = grid.classify(coordinates(rows), include_hierarchy=True)

        for key, path in zip(chunk_keys, labels):
            metadata[key][HIERARCHY_LABELS] = tuple(int(value) for value in path)

        if cached_labels is not None:
            cached_labels[start:start + len(chunk_keys)] = labels

        progress.update(len(chunk_keys))

    progress.close()

    if cache_path is not None:
        _save_hierarchy_labels(metadata, grid, cached_labels, cache_path)
