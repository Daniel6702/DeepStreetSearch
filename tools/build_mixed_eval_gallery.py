"""Create a reproducible 200k evaluation gallery without copying panorama archives.

The external test set alone has fewer than 200k panoramas. Include *all* test
panoramas, then sample training panoramas as gallery distractors. Both the 20k
and 200k indices can be drawn from this one gallery; paired queries are still
sampled exclusively from the separate external test dataset.

Example:
  python -m tools.build_mixed_eval_gallery --train .../world_train \
      --test .../world_test --count 200000 --output outputs/eval_galleries/mixed_200000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_id(row: dict[str, str]) -> str:
    source = row.get("image_path", "")
    key = Path(source.split("::", 1)[-1]).stem if source else row.get("panoid", "")
    return key[:-4] if key.endswith("_360") else key


def read_metadata(root: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with (root / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        if not columns:
            raise ValueError(f"Missing metadata columns: {root}")
        records = {}
        for row in reader:
            key = image_id(row)
            if not key or key in records:
                raise ValueError(f"Missing or repeated panorama ID {key!r} in {root}")
            records[key] = row
    return columns, records


def shard_aliases(root: Path, label: str) -> dict[str, tuple[str, Path]]:
    archives = sorted((*root.glob("images/shard_*.tar"), *root.glob("images/shard_*.zip")))
    if not archives:
        raise FileNotFoundError(f"No shard archives in {root / 'images'}")
    return {
        archive.name: (f"shard_{label}_{archive.name[len('shard_'):]}", archive.resolve())
        for archive in archives
    }


def prepare_rows(records, aliases):
    prepared = []
    for key, row in records:
        row = dict(row)
        if row.get("image_path"):
            source, separator, member = row["image_path"].partition("::")
            original = Path(source).name
            if original not in aliases:
                raise ValueError(f"Missing archive {original} for panorama {key}")
            row["image_path"] = aliases[original][0] + (separator + member if separator else "")
        prepared.append(row)
    return prepared


def make_gallery(train: Path, test: Path, output: Path, count: int, seed: int) -> dict:
    train, test = train.resolve(), test.resolve()
    if train == test:
        raise ValueError("Training and external-test dataset directories must differ")
    train_columns, training = read_metadata(train)
    test_columns, testing = read_metadata(test)
    overlap = set(training) & set(testing)
    if overlap:
        raise ValueError(f"Training and external test overlap by {len(overlap)} panorama IDs")
    if not len(testing) <= count <= len(training) + len(testing):
        raise ValueError(f"Gallery count must be between {len(testing)} and {len(testing) + len(training)}")

    train_count = count - len(testing)
    selected_training = sorted(random.Random(seed).sample(sorted(training), train_count))
    test_aliases = shard_aliases(test, "test")
    train_aliases = shard_aliases(train, "train")
    rows = prepare_rows(sorted(testing.items()), test_aliases)
    rows += prepare_rows(((key, training[key]) for key in selected_training), train_aliases)
    columns = list(dict.fromkeys([*test_columns, *train_columns]))

    metadata_text = io.StringIO(newline="")
    writer = csv.DictWriter(metadata_text, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    metadata_bytes = metadata_text.getvalue().encode("utf-8")
    metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
    manifest = {
        "count": count,
        "seed": seed,
        "test_count": len(testing),
        "train_count": train_count,
        "test_dataset": str(test),
        "train_dataset": str(train),
        "test_metadata_sha256": sha256(test / "metadata.csv"),
        "train_metadata_sha256": sha256(train / "metadata.csv"),
        "gallery_metadata_sha256": metadata_digest,
    }

    output.mkdir(parents=True, exist_ok=True)
    metadata_path, manifest_path = output / "metadata.csv", output / "manifest.json"
    if metadata_path.exists() and sha256(metadata_path) != metadata_digest:
        raise ValueError(f"Existing gallery metadata differs: {metadata_path}; choose another output")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Existing gallery manifest differs: {manifest_path}; choose another output")

    images_dir = output / "images"
    images_dir.mkdir(exist_ok=True)
    for aliases in (test_aliases, train_aliases):
        for alias, target in aliases.values():
            destination = images_dir / alias
            if destination.is_symlink():
                if destination.resolve() != target:
                    raise ValueError(f"Gallery symlink points elsewhere: {destination}")
            elif destination.exists():
                raise ValueError(f"Gallery contains a non-symlink archive: {destination}")
            else:
                destination.symlink_to(target)

    if not metadata_path.exists():
        temporary = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary.write_bytes(metadata_bytes)
        os.replace(temporary, metadata_path)
    if not manifest_path.exists():
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1 or args.seed < 0:
        parser.error("--count must be positive and --seed nonnegative")
    summary = make_gallery(args.train, args.test, args.output, args.count, args.seed)
    print(f"Experiment: evaluation gallery: {summary['count']:,} panoramas "
          f"({summary['test_count']:,} external + {summary['train_count']:,} training distractors); "
          f"saved to {args.output}")


if __name__ == "__main__":
    main()
