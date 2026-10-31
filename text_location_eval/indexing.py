"""Build a reproducible, checkpoint-specific Street View embedding gallery."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

from .common import file_sha256


def expected_signature(args) -> dict:
    return {
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "grid_sha256": file_sha256(args.grid),
        "metadata_sha256": file_sha256(Path(args.dataset) / "metadata.csv"),
        "dataset": str(Path(args.dataset).resolve()),
        "model": args.model,
        "seed": args.seed,
        "num_images": args.num_images,
    }


def load_index(path: str | Path) -> dict:
    import torch

    index = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(index, dict) or not {"signature", "records", "embeddings"} <= index.keys():
        raise ValueError(f"Invalid reference index: {path}")
    if index["embeddings"].ndim != 3 or index["embeddings"].shape[1] != 6:
        raise ValueError("Index embeddings must have shape [panoramas, 6, dimensions]")
    if len(index["records"]) != index["embeddings"].shape[0]:
        raise ValueError("Reference index records and embeddings differ in length")
    return index


def _selected_keys(metadata: dict, count: int, seed: int) -> set[str]:
    """Match PanoramaDataset's seeded sampling exactly."""
    return (set(metadata) if count == len(metadata)
            else set(random.Random(seed).sample(list(metadata), count)))


def _load_parts(torch, directory: Path, signature: dict,
                selected: set[str]) -> tuple[list[Path], set[str]]:
    parts = sorted(directory.glob("part-*.pt")) if directory.exists() else []
    completed = set()
    for number, path in enumerate(parts):
        if path.name != f"part-{number:06d}.pt":
            raise ValueError(f"Index progress has missing or unexpected part numbers in {directory}")
        part = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(part, dict) or part.get("signature") != signature:
            raise ValueError(f"Index progress belongs to different settings: {path}")
        rows, vectors = part.get("records"), part.get("embeddings")
        if (not isinstance(rows, list) or vectors is None or vectors.ndim != 3
                or vectors.shape[1] != 6 or len(rows) != vectors.shape[0]):
            raise ValueError(f"Invalid saved index part: {path}")
        ids = [row["image_id"] for row in rows]
        if len(set(ids)) != len(ids) or completed.intersection(ids):
            raise ValueError(f"Duplicate panorama IDs in saved part: {path}")
        if not set(ids).issubset(selected):
            raise ValueError(f"Saved part contains panoramas outside this sample: {path}")
        completed.update(ids)
        del part
    return parts, completed


def _save_part(torch, path: Path, signature: dict,
               records: list[dict], embeddings: list) -> None:
    """Commit a finished chunk atomically; interrupted .tmp files are ignored."""
    if not records or not embeddings:
        raise ValueError("Cannot save an empty index part")
    data = {"signature": signature, "records": records,
            "embeddings": torch.cat(embeddings, dim=0)}
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def build_index(args) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoProcessor

    from adaptive_geogrid import load_grid
    from config import NUM_VIEWS
    from modules.dataset import PanoramaDataset
    from modules.dataset.shards import SOURCE_SHARD, image_key
    from modules.model import GeoSearchModel

    output = Path(args.output)
    signature = expected_signature(args)
    save_every = getattr(args, "save_every", 5000)
    if save_every < 1:
        raise ValueError("--save-every must be positive")
    parts_dir = output.with_name(f"{output.stem}.parts")
    if output.is_file() and not args.rebuild:
        existing = load_index(output)
        if existing["signature"] != signature:
            raise ValueError("Reference index is from different data, settings, grid, or checkpoint. Use --rebuild.")
        print(f"Existing index valid: {len(existing['records']):,} panoramas in {output}")
        return

    metadata = PanoramaDataset.load_metadata(args.dataset)
    count = min(args.num_images, len(metadata))
    if count <= 0:
        raise ValueError("No panoramas available")
    selected = _selected_keys(metadata, count, args.seed)
    if args.rebuild and parts_dir.exists():
        old_parts = list(parts_dir.glob("part-*.pt"))
        for path in old_parts:
            path.unlink()
        print(f"Rebuild requested: removed {len(old_parts)} saved index parts from {parts_dir}")
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths, completed = _load_parts(torch, parts_dir, signature, selected)
    if completed:
        print(f"Resuming index: {len(completed):,}/{count:,} panoramas already saved "
              f"in {len(part_paths)} chunks")
    remaining = {key: row for key, row in metadata.items()
                 if key in selected and key not in completed}
    del metadata, selected

    if remaining:
        dataset = PanoramaDataset(args.dataset, metadata=remaining, fraction=1.0,
                                  seed=args.seed, shuffle=False)
        # With fraction=1, PanoramaDataset normally reads every image in each
        # shard. Restrict decoding to unfinished keys, including on the first
        # run when our metadata is already the selected subset.
        dataset.selected_keys = set(remaining)
        loader = dataset.loader(batch_size=args.batch_size, num_workers=args.workers,
                                prepare_panorama_crops=True, pin_memory=False)

        grid = load_grid(args.grid)
        model = GeoSearchModel(grid, model_name=args.model).to(args.device).eval()
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        processor = AutoProcessor.from_pretrained(args.model)

        embeddings, records = [], []
        processed = len(completed)
        with torch.inference_mode():
            for crops_batch, metadata_batch in tqdm(loader, desc="Indexing panoramas",
                                                      total=math.ceil(len(remaining) / args.batch_size)):
                try:
                    if any(len(crops) != NUM_VIEWS for crops in crops_batch):
                        raise ValueError("Unexpected number of panorama crops")
                    views = [crop.image for crops in crops_batch for crop in crops]
                    inputs = processor(images=views, return_tensors="pt")
                    inputs = {k: v.to(args.device) for k, v in inputs.items() if torch.is_tensor(v)}
                    _, features = model.encode_images(inputs, len(crops_batch), NUM_VIEWS)
                    embeddings.append(features.reshape(len(crops_batch), NUM_VIEWS, -1).half().cpu())

                    for row in metadata_batch:
                        image_path = row.get("image_path", "")
                        key = image_key(image_path.split("::")[-1]) if image_path else row.get("panoid")
                        lat = row.get("lat") or row.get("pano_lat")
                        lon = row.get("lon") or row.get("pano_lon")
                        if not key or lat is None or lon is None:
                            raise ValueError(f"Missing panorama id/coordinates in metadata: {row}")
                        records.append({
                            "image_id": key,
                            "latitude": float(lat),
                            "longitude": float(lon),
                            "shard": row[SOURCE_SHARD],
                        })
                finally:
                    for crops in crops_batch:
                        for crop in crops:
                            crop.image.close()
                if len(records) >= save_every:
                    part_path = parts_dir / f"part-{len(part_paths):06d}.pt"
                    _save_part(torch, part_path, signature, records, embeddings)
                    part_paths.append(part_path)
                    processed += len(records)
                    print(f"Saved index progress: {processed:,}/{count:,} panoramas")
                    records, embeddings = [], []

        if records:
            part_path = parts_dir / f"part-{len(part_paths):06d}.pt"
            _save_part(torch, part_path, signature, records, embeddings)
            part_paths.append(part_path)
            processed += len(records)
            print(f"Saved index progress: {processed:,}/{count:,} panoramas")

    # Merge the saved parts only after every panorama is safely on disk.
    all_embeddings, all_records = [], []
    for path in tqdm(part_paths, desc="Combining saved index parts"):
        part = torch.load(path, map_location="cpu", weights_only=True)
        all_embeddings.append(part["embeddings"])
        all_records.extend(part["records"])
    if len(all_records) != count or len({r["image_id"] for r in all_records}) != count:
        raise RuntimeError(f"Expected {count} unique panoramas, saved {len(all_records)}")
    index = {"signature": signature, "records": all_records,
             "embeddings": torch.cat(all_embeddings, dim=0)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(index, temporary)
    os.replace(temporary, output)
    print(f"Saved {count:,} panoramas / {count * NUM_VIEWS:,} crops to {output}")
