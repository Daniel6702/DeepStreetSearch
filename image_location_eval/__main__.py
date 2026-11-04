"""Evaluate image-to-location predictions from the geographic head."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from image_location_eval.metrics import calculate_metrics


def parse_args():
    from config import DATASET_PATH, SIGLIP_MODEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grid", default="outputs/grids/world.aggrid")
    parser.add_argument("--dataset", default=f"{DATASET_PATH}/world", help="Images to evaluate")
    parser.add_argument("--train-dataset", default=f"{DATASET_PATH}/world", help="Dataset used for training")
    parser.add_argument("--split", choices=("heldout", "external", "all"), default="heldout",
                        help="heldout: exclude checkpoint training sample; external: distinct test dataset; all: in-sample diagnostic")
    parser.add_argument("--num-images", type=int, default=2000, help="Limit examples after split (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic evaluation subset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=SIGLIP_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, help="Directory for summary.json and predictions.csv")
    args = parser.parse_args()
    if args.num_images < 0 or args.batch_size < 1 or args.workers < 0:
        parser.error("--num-images and --workers must be non-negative; --batch-size must be positive")
    return args


def choose_metadata(args, checkpoint_config):
    """Reproduce training's exact seeded subset before selecting held-out IDs."""
    from modules.dataset import PanoramaDataset

    eval_metadata = PanoramaDataset.load_metadata(args.dataset)
    if args.split == "heldout":
        if Path(args.dataset).resolve() != Path(args.train_dataset).resolve():
            raise ValueError("Use --split external when --dataset differs from --train-dataset")
        fraction = checkpoint_config.get("dataset_fraction")
        seed = checkpoint_config.get("seed")
        if fraction is None or seed is None or not 0 < float(fraction) < 1:
            print("WARNING: No held-out split is available for this checkpoint. "
                  "Evaluating training images (in-sample results).")
            args.split = "all"
            eligible = list(eval_metadata)
        else:
            training_count = max(1, int(len(eval_metadata) * float(fraction)))
            # PanoramaDataset selects from list(metadata), using random.Random(seed).sample.
            training_keys = set(random.Random(int(seed)).sample(list(eval_metadata), training_count))
            eligible = [key for key in eval_metadata if key not in training_keys]
    elif args.split == "external":
        if Path(args.dataset).resolve() == Path(args.train_dataset).resolve():
            raise ValueError("External evaluation requires a distinct --dataset and --train-dataset")
        # Prevent accidentally evaluating duplicate panoramas from the training dataset.
        training_ids = set(PanoramaDataset.load_metadata(args.train_dataset))
        overlap = training_ids.intersection(eval_metadata)
        if overlap:
            raise ValueError(f"External test dataset overlaps training dataset by {len(overlap)} panorama IDs")
        eligible = list(eval_metadata)
    else:
        print("WARNING: --split all includes training images. Results are NOT held-out evaluation.")
        eligible = list(eval_metadata)

    if not eligible:
        raise ValueError("No panoramas remain in the evaluation split")
    if args.num_images and args.num_images < len(eligible):
        chosen = set(random.Random(args.seed).sample(eligible, args.num_images))
    else:
        chosen = set(eligible)
    return {key: value for key, value in eval_metadata.items() if key in chosen}, len(eligible)


def grid_coordinates(grid):
    tiles = grid.level(0).sort_values("tile_id")
    ids = tiles["tile_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids, np.arange(grid.n_classes[0])):
        raise ValueError("Fine grid tile IDs must be contiguous and zero-based")
    points = tiles.geometry.representative_point()
    return np.column_stack((points.x.to_numpy(), points.y.to_numpy()))


def main():
    import torch
    from tqdm import tqdm
    from transformers import AutoProcessor
    from adaptive_geogrid import load_grid
    from config import NUM_VIEWS
    from modules.dataset import PanoramaDataset
    from modules.dataset.metadata import HIERARCHY_LABELS, coordinates, precompute_hierarchy_labels
    from modules.dataset.collators import close_prepared_crops
    from modules.dataset.shards import image_key
    from modules.model import GeoSearchModel

    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    checkpoint_hash = digest.hexdigest()[:12]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config") or {}
    metadata, eligible_count = choose_metadata(args, config)
    print(f"Evaluating {len(metadata):,} of {eligible_count:,} {args.split} panoramas")

    grid = load_grid(args.grid)
    precompute_hierarchy_labels(metadata, grid)  # Only test metadata; never rewrite training label cache.
    hierarchy = grid.hierarchy_table().to_numpy(dtype=np.int64, copy=True)
    centers = grid_coordinates(grid)
    dataset = PanoramaDataset(args.dataset, metadata=metadata, shuffle=False, fraction=1.0)
    dataset.selected_keys = set(metadata)
    loader = dataset.loader(batch_size=args.batch_size, num_workers=args.workers,
                            prepare_panorama_crops=True)

    model = GeoSearchModel(
        grid, model_name=args.model,
        geo_head=config.get("geo_head", "linear"),
        geo_pooling=config.get("geo_pooling", "mean"),
        geo_view_dropout=config.get("geo_view_dropout", 0.0),
    ).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    processor = AutoProcessor.from_pretrained(args.model)

    actual_coordinates, actual_paths, top_classes, rows = [], [], [], []
    outside = 0
    max_k = min(10, grid.n_classes[0])
    with torch.inference_mode():
        for crops_batch, metadata_batch in tqdm(loader, total=math.ceil(len(metadata) / args.batch_size),
                                                 desc="Image-to-location eval"):
            try:
                labels = np.asarray([row[HIERARCHY_LABELS] for row in metadata_batch], dtype=np.int64)
                valid = labels[:, 0] >= 0
                outside += int((~valid).sum())
                if not valid.any():
                    continue
                if any(len(crops) != NUM_VIEWS for crops in crops_batch):
                    raise ValueError("Unexpected number of panorama crops")
                images = [crop.image for crops in crops_batch for crop in crops]
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items() if torch.is_tensor(v)}
                with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                    enabled=torch.device(args.device).type == "cuda"):
                    logits, _ = model.encode_images(inputs, len(crops_batch), NUM_VIEWS)
                top = torch.topk(logits, k=max_k, dim=1).indices.cpu().numpy()
                actual_coordinates.extend(coordinates(metadata_batch)[valid])
                actual_paths.extend(labels[valid])
                top_classes.extend(top[valid])
                rows.extend(row for row, keep in zip(metadata_batch, valid) if keep)
            finally:
                close_prepared_crops(crops_batch)

    if not rows:
        raise ValueError("No evaluated panoramas fall within the geographic grid")
    actual_coordinates = np.asarray(actual_coordinates, dtype=np.float64)
    actual_paths = np.asarray(actual_paths, dtype=np.int64)
    top_classes = np.asarray(top_classes, dtype=np.int64)
    metrics, distances = calculate_metrics(actual_coordinates, actual_paths, top_classes, hierarchy, centers)
    summary = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256_12": checkpoint_hash,
        "grid": args.grid, "dataset": args.dataset, "split": args.split,
        "seed": args.seed, "eligible_count": eligible_count,
        "selected_count": len(metadata), "skipped_outside_grid": outside,
        "geo_head": config.get("geo_head", "linear"),
        "geo_pooling": config.get("geo_pooling", "mean"),
        **metrics,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("panoid", "latitude", "longitude", "true_fine_cell", "pred_fine_cell",
                         "pred_latitude", "pred_longitude", "error_km", "top5_hit", "top10_hit",
                         "best_top5_error_km", "best_top10_error_km"))
        for i, row in enumerate(rows):
            top5, top10 = min(5, max_k), min(10, max_k)
            predicted = centers[top_classes[i, 0]]
            writer.writerow((row.get("panoid", image_key(row.get("image_path", ""))),
                             actual_coordinates[i, 1], actual_coordinates[i, 0],
                             actual_paths[i, 0], top_classes[i, 0], predicted[1], predicted[0],
                             distances[i, 0], int(actual_paths[i, 0] in top_classes[i, :top5]),
                             int(actual_paths[i, 0] in top_classes[i, :top10]),
                             np.min(distances[i, :top5]), np.min(distances[i, :top10])))

    median = metrics["median_error_km"]
    mean = metrics["mean_error_km"]
    print(f"Median: {median:.1f} km | Mean: {mean:.1f} km" if median is not None and mean is not None
          else "No valid geographic distances for this sample")
    if metrics["invalid_distance_count"]:
        print(f"WARNING: {metrics['invalid_distance_count']} predicted cell coordinates were invalid; "
              "distance statistics exclude them (accuracy counts them as misses).")
    print("Within distance:", ", ".join(f"{km} km: {value:.1%}"
          for km, value in metrics["accuracy_within_km"].items()))
    print("Grid hierarchy top-1:", ", ".join(f"L{i}: {value:.1%}"
          for i, value in enumerate(metrics["hierarchy_top1_accuracy"])))
    print("Fine cell top-k:", ", ".join(f"@{k}: {value:.1%}"
          for k, value in metrics["fine_cell_topk_accuracy"].items()))
    print(f"Saved: {output / 'summary.json'} and {output / 'predictions.csv'}")


if __name__ == "__main__":
    main()
