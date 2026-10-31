
    
  
"""Inspect one retrieved location as its original full panorama (no GPU needed)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_run_dir(run_dir: str | Path | None) -> Path:
    if run_dir is None:
        candidates = list((PROJECT_ROOT / "outputs/eval/text_location_eval").glob("*/retrieval.json"))
        if not candidates:
            raise FileNotFoundError("No retrieval.json found. Set RUN_DIR to your evaluation output folder.")
        return max(candidates, key=lambda path: path.stat().st_mtime).parent
    path = Path(run_dir).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not (path / "retrieval.json").is_file():
        raise FileNotFoundError(f"Missing retrieval.json in {path}")
    return path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def original_panorama(run_dir: Path, record: dict) -> Path:
    """Extract the original 360° panorama once and cache it for notebook use."""
    from PIL import Image

    image_id = record["image_id"]
    key = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:20]
    cache_dir = run_dir / "original_panoramas"
    target = cache_dir / f"{key}.jpg"
    if target.is_file():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)

    shard = Path(record["shard"])
    if not shard.is_file():
        raise FileNotFoundError(f"Original panorama shard unavailable: {shard}")

    def matches(filename: str) -> bool:
        name = Path(filename)
        if name.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            return False
        stem = name.stem
        return (stem[:-4] if stem.endswith("_360") else stem) == image_id

    def save_stream(stream) -> None:
        temporary = target.with_suffix(".tmp")
        try:
            with Image.open(stream) as image:
                with image.convert("RGB") as rgb:
                    rgb.save(temporary, format="JPEG", quality=95)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    if shard.suffix.lower() == ".tar":
        with tarfile.open(shard, "r:*") as archive:
            for member in archive:
                if member.isfile() and matches(member.name):
                    stream = archive.extractfile(member)
                    if stream is not None:
                        with stream:
                            save_stream(stream)
                        return target
    elif shard.suffix.lower() == ".zip":
        with zipfile.ZipFile(shard) as archive:
            for filename in archive.namelist():
                if matches(filename):
                    with archive.open(filename) as stream:
                        save_stream(stream)
                    return target
    else:
        raise ValueError(f"Unsupported dataset shard format: {shard.suffix}")
    raise FileNotFoundError(f"Could not find panorama {image_id!r} in {shard}")


def inspect(run_dir: str | Path | None, query_id: str, rank: int = 1,
            save_path: str | Path | None = None, show: bool = True) -> None:
    """Display precisely one original panorama and its matching saved judgment."""
    if rank < 1:
        raise ValueError("Rank is 1-based: 1 = best result, 2 = second-best, etc.")
    import matplotlib.pyplot as plt
    from PIL import Image

    run_dir = find_run_dir(run_dir)
    with (run_dir / "retrieval.json").open(encoding="utf-8") as stream:
        retrieval = json.load(stream)
    query = next((row for row in retrieval["queries"] if row["id"] == query_id), None)
    if query is None:
        raise ValueError(f"Unknown query ID: {query_id}")
    if rank > len(query["matches"]):
        raise ValueError(f"{query_id} has only {len(query['matches'])} retrieved results")
    record = query["matches"][rank - 1]
    judgment = next((row for row in load_jsonl(run_dir / "judgments.jsonl")
                     if row["query_id"] == query_id and row["image_id"] == record["image_id"]),
                    None)
    rubric = next((row["categories"] for row in load_jsonl(run_dir / "query_analysis.jsonl")
                   if row["query_id"] == query_id), {})

    print(f"Query {query_id}: {query['query']}")
    print(f"Result #{rank} / {len(query['matches'])} | Similarity: {record['similarity']:.4f}")
    print(f"Coordinates: {record['latitude']:.6f}, {record['longitude']:.6f}")
    print("Map: https://www.openstreetmap.org/"
          f"?mlat={record['latitude']}&mlon={record['longitude']}"
          f"#map=15/{record['latitude']}/{record['longitude']}")
    if judgment is None:
        print("Qwen judgment: MISSING (this pair has not been successfully judged)")
    else:
        print(f"Qwen holistic relevance: {judgment['holistic_score']}/3")
        print(f"Qwen summary: {judgment.get('summary', '')}")
        print("Category scores:")
        for name, data in judgment.get("category_scores", {}).items():
            score = data["score"] if isinstance(data, dict) else data
            evidence = data.get("evidence", "") if isinstance(data, dict) else ""
            weight = rubric.get(name, {}).get("importance", "?")
            print(f"  {name}: {score}/3 (importance {weight})"
                  + (f" — {evidence}" if evidence else ""))

    path = original_panorama(run_dir, record)
    fig, ax = plt.subplots(figsize=(16, 8))
    with Image.open(path) as image:
        ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{query_id} — retrieved rank #{rank}", fontsize=14)
    fig.tight_layout()
    if save_path is not None:
        destination = Path(save_path).expanduser()
        if not destination.is_absolute():
            destination = PROJECT_ROOT / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=120, bbox_inches="tight")
        print(f"Saved {destination}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Evaluation output directory (default: latest run)")
    parser.add_argument("--query-id", required=True, help="Example: Q011")
    parser.add_argument("--rank", type=int, default=1, help="1 = best, 2 = second-best")
    parser.add_argument("--save-path", help="Optional output PNG file")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot")
    args = parser.parse_args()
    inspect(args.run_dir, args.query_id, args.rank, args.save_path, not args.no_show)


if __name__ == "__main__":
    main()

