"""Lock the same external paired positives for 20k and 200k gallery tests."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def choose_ids(records: list[dict], external_ids: set[str], count: int, seed: int) -> list[str]:
    available = sorted({row["image_id"] for row in records if row["image_id"] in external_ids})
    if len(available) < count:
        raise ValueError(f"Only {len(available)} external positives in gallery, need {count}")
    return sorted(random.Random(seed).sample(available, count))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--external-dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1 or args.seed < 0:
        parser.error("--count must be positive and --seed nonnegative")

    from modules.dataset import PanoramaDataset
    from modules.inference import load_reference_index

    index = load_reference_index(args.index)
    external_ids = set(PanoramaDataset.load_metadata(args.external_dataset))
    selected = choose_ids(index["records"], external_ids, args.count, args.seed)
    content = "".join(image_id + "\n" for image_id in selected)
    if args.output.exists():
        if args.output.read_text() != content:
            raise ValueError(f"Existing paired positive selection differs: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(f"Experiment: locked {len(selected)} external paired-positive panoramas: {args.output}")


if __name__ == "__main__":
    main()
