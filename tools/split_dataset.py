"""Create a disjoint train/test metadata split without copying image archives.

Preview (read-only):
    python tools/split_dataset.py --source /path/to/datasets/world
Create datasets/world_train and datasets/world_test:
    python tools/split_dataset.py --source /path/to/datasets/world --apply

Original data is never modified. Each split links to the original archives and
per-shard description databases; only metadata CSVs are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path


def _read_source_ids(path: Path) -> dict[str, tuple[str, str]]:
    ids = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"panoid", "pano_lat", "pano_lon"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing panoid/pano_lat/pano_lon columns: {path}")
        for line, row in enumerate(reader, start=2):
            panoid = (row["panoid"] or "").strip()
            if not panoid or panoid in ids:
                raise ValueError(f"Missing or duplicate panoid at {path}:{line}: {panoid!r}")
            ids[panoid] = (row["pano_lat"], row["pano_lon"])
    if len(ids) < 2:
        raise ValueError(f"Need at least two panoramas in {path}")
    return ids


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, int]]:
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def _write_csv_split(source: Path, train: Path | None, test: Path | None,
                     test_ids: set[str], source_ids: dict, archives: set[str] | None = None,
                     require_image_paths: bool = False) -> tuple[int, int]:
    seen = set()
    counts = [0, 0]
    with ExitStack() as stack:
        stream = stack.enter_context(source.open(newline="", encoding="utf-8"))
        reader = csv.DictReader(stream)
        if "panoid" not in (reader.fieldnames or []):
            raise ValueError(f"Missing panoid column: {source}")
        if require_image_paths and "image_path" not in reader.fieldnames:
            raise ValueError(f"Missing image_path column: {source}")
        writers = [None, None]
        for idx, path in enumerate((train, test)):
            if path is not None:
                output = stack.enter_context(path.open("w", newline="", encoding="utf-8"))
                writers[idx] = csv.DictWriter(output, fieldnames=reader.fieldnames)
                writers[idx].writeheader()

        for line, row in enumerate(reader, start=2):
            panoid = (row["panoid"] or "").strip()
            if not panoid or panoid not in source_ids:
                raise ValueError(f"Unknown or missing panoid in {source}:{line}: {panoid!r}")
            if panoid in seen:
                raise ValueError(f"Duplicate panoid in {source}:{line}: {panoid}")
            seen.add(panoid)
            if require_image_paths:
                image_path = row.get("image_path", "") or ""
                archive, separator, filename = image_path.partition("::")
                stem = Path(filename).stem
                image_id = stem[:-4] if stem.endswith("_360") else stem
                if not separator or not filename or image_id != panoid:
                    raise ValueError(f"Unexpected image_path in {source}:{line}: {image_path!r}")
                if Path(archive).name not in archives:
                    raise ValueError(f"Image archive missing for {panoid}: {archive}")
                if (row["pano_lat"], row["pano_lon"]) != source_ids[panoid]:
                    raise ValueError(f"Coordinate mismatch for {panoid} between source metadata CSVs")
            idx = 1 if panoid in test_ids else 0
            counts[idx] += 1
            if writers[idx] is not None:
                writers[idx].writerow(row)

    if require_image_paths and seen != source_ids.keys():
        raise ValueError(f"metadata_full.csv is missing {len(source_ids.keys() - seen)} panorama IDs")
    return counts[0], counts[1]


def _link_split_assets(stage: Path, source: Path, final: Path) -> None:
    for name in ("images", "descriptions"):
        if not (source / name).is_dir():
            raise FileNotFoundError(f"Missing required dataset directory: {source / name}")
        # Both stage and final directories are siblings of source.
        (stage / name).symlink_to(os.path.relpath(source / name, final), target_is_directory=True)
    if (source / "descriptions.sqlite").is_file():
        (stage / "descriptions.sqlite").symlink_to(
            os.path.relpath(source / "descriptions.sqlite", final))
    (stage / "metadata_full.csv").symlink_to("metadata.csv")


def split(source: Path, test_fraction: float, seed: int, apply: bool) -> dict:
    source = source.resolve(strict=True)
    if not 0 < test_fraction < 1:
        raise ValueError("--test-fraction must be in (0, 1)")
    if seed < 0:
        raise ValueError("--seed must be nonnegative")

    train = source.with_name(source.name + "_train")
    test = source.with_name(source.name + "_test")
    for destination in (train, test):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing dataset: {destination}")

    required = [source / name for name in
                ("metadata.csv", "metadata_full.csv", "metadata_geocoded.csv")]
    before = _snapshot(required)
    ids = _read_source_ids(required[0])
    count = max(1, min(len(ids) - 1, int(len(ids) * test_fraction)))
    test_ids = set(random.Random(seed).sample(sorted(ids), count))
    archives = {path.name for path in (source / "images").glob("shard_*.zip")}
    archives.update(path.name for path in (source / "images").glob("shard_*.tar"))
    if not archives:
        raise FileNotFoundError(f"No image archives found in {source / 'images'}")
    if not (source / "descriptions").is_dir():
        raise FileNotFoundError(f"No description directory found in {source / 'descriptions'}")

    digest = hashlib.sha256()
    for panoid in sorted(test_ids):
        digest.update(panoid.encode("utf-8") + b"\n")
    manifest = {
        "source": str(source), "seed": seed, "test_fraction": test_fraction,
        "train_panoramas": len(ids) - count, "test_panoramas": count,
        "test_panoids_sha256": digest.hexdigest(),
        "note": "Split metadata only; images and descriptions are shared read-only via symlinks",
    }

    train_stage = test_stage = None
    try:
        if apply:
            # Stage as sibling directories so relative symlinks remain valid on rename.
            train_stage = Path(tempfile.mkdtemp(prefix=f".{train.name}-staging-", dir=source.parent))
            test_stage = Path(tempfile.mkdtemp(prefix=f".{test.name}-staging-", dir=source.parent))
        full_counts = _write_csv_split(
            required[1], train_stage / "metadata.csv" if apply else None,
            test_stage / "metadata.csv" if apply else None,
            test_ids, ids, archives, require_image_paths=True)
        geo_counts = _write_csv_split(
            required[2], train_stage / "metadata_geocoded.csv" if apply else None,
            test_stage / "metadata_geocoded.csv" if apply else None,
            test_ids, ids)
        if full_counts != (len(ids) - count, count):
            raise ValueError(f"Incorrect full metadata split counts: {full_counts}")
        if before != _snapshot(required):
            raise RuntimeError("Source metadata changed during splitting; retry when writes have stopped")
        manifest["train_geocoded_rows"], manifest["test_geocoded_rows"] = geo_counts

        if apply:
            for stage, final in ((train_stage, train), (test_stage, test)):
                _link_split_assets(stage, source, final)
                (stage / "split_manifest.json").write_text(
                    json.dumps({**manifest, "split": final.name}, indent=2) + "\n",
                    encoding="utf-8")
            # No destination overwrite: rename only after all checks/writes succeed.
            if train.exists() or test.exists():
                raise FileExistsError("Output destination appeared while preparing split")
            train_stage.rename(train)
            try:
                test_stage.rename(test)
            except Exception:
                train.rename(train_stage)  # Roll back first published directory.
                raise
    finally:
        for stage in (train_stage, test_stage):
            if stage is not None and stage.exists():
                shutil.rmtree(stage)

    print(json.dumps({**manifest, "train_dataset": str(train),
                      "test_dataset": str(test), "created": apply}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true", help="Create split datasets (default: validate/preview only)")
    args = parser.parse_args()
    split(args.source, args.test_fraction, args.seed, args.apply)


if __name__ == "__main__":
    main()
