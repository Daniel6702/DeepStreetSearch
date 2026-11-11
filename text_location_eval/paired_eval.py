"""Objective text-to-image retrieval: a caption's source panorama is the positive.

Usage: python -m text_location_eval.paired_eval --checkpoint ... --index ...
This does not use Qwen, subjective judgments, or the geographic head.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from config import DATASET_PATH, NUM_VIEWS, SIGLIP_MODEL
from text_location_eval.common import write_json


def select_records(records, metadata, config, count: int, seed: int,
                   split_mode: str = "auto", external_train_ids=None):
    """Choose gallery positives, preferring held-out IDs when the index contains any.

    The index builder and training sampler may share a seed. In that case the
    entire smaller index can be a subset of the training IDs, so held-out
    evaluation is impossible *with this index*. Auto mode reports the fallback
    to in-sample evaluation instead of crashing after an expensive index run.
    """
    if split_mode not in {"auto", "heldout", "in_sample", "external"}:
        raise ValueError(f"Unknown split mode: {split_mode}")

    fraction, train_seed = config.get("dataset_fraction"), config.get("seed")
    matched = sorted(
        (record for record in records if record["image_id"] in metadata),
        key=lambda record: record["image_id"],
    )
    if not matched:
        raise ValueError("No gallery panoramas have matching evaluation metadata; check --dataset")

    if split_mode == "external":
        if external_train_ids is None:
            raise ValueError("External evaluation requires training panorama IDs")
        overlap = set(metadata).intersection(external_train_ids)
        if overlap:
            raise ValueError(f"External test dataset overlaps training by {len(overlap)} panorama IDs")
        available, split = matched, "external"
    elif fraction is not None and train_seed is not None and 0 < float(fraction) < 1:
        training_count = max(1, int(len(metadata) * float(fraction)))
        training_ids = set(random.Random(int(train_seed)).sample(list(metadata), training_count))
        heldout = [record for record in matched if record["image_id"] not in training_ids]
        in_sample = [record for record in matched if record["image_id"] in training_ids]
    else:
        heldout, in_sample = [], matched

    if split_mode == "external":
        pass  # Already validated against the separate training dataset.
    elif split_mode == "heldout":
        available, split = heldout, "heldout"
    elif split_mode == "in_sample":
        available, split = in_sample, "in_sample"
    elif heldout:
        available, split = heldout, "heldout"
    else:
        available, split = in_sample, "in_sample"
        print("WARNING: No held-out panoramas in this index; evaluating in-sample. "
              "For a held-out paired evaluation, build an index using a different sampling seed.")

    if not available:
        raise ValueError(f"No {split} panoramas available in the index")
    if count and count < len(available):
        positions = sorted(random.Random(seed).sample(range(len(available)), count))
        available = [available[position] for position in positions]
    return available, split


def summarize_rows(rows, max_k: int):
    def values(items):
        if not items:
            return None
        result = {"queries": len(items), "truncated_fraction": sum(r["truncated"] for r in items) / len(items)}
        for k in (1, 5, 10):
            result[f"recall@{k}"] = sum(0 < r["rank"] <= min(k, max_k) for r in items) / len(items)
        result[f"mrr@{max_k}"] = sum(1 / r["rank"] for r in items if r["rank"]) / len(items)
        return result

    return {
        "overall": values(rows),
        "by_prompt": {str(i): values([row for row in rows if row["prompt"] == i]) for i in range(1, 5)},
    }


def load_captions(selected, metadata, descriptions: Path, prompt_indices, seed: int):
    from modules.descriptions import DescriptionStore

    rng = random.Random(seed)
    stores = {}
    rows = []
    missing = 0
    try:
        for record in selected:
            panoid = metadata[record["image_id"]].get("panoid") or record["image_id"]
            crop = rng.randrange(NUM_VIEWS)
            database = (descriptions if descriptions.is_file()
                        else descriptions / (Path(record["shard"]).name + ".sqlite"))
            if not database.is_file():
                missing += 1
                continue
            if database not in stores:
                stores[database] = DescriptionStore(database, read_only=True)
            captions = stores[database].get_all(panoid, crop)
            for prompt in prompt_indices:
                caption = captions.get(prompt - 1)
                if caption and caption.strip():
                    rows.append({"image_id": record["image_id"], "crop": crop,
                                 "prompt": prompt, "caption": caption})
                else:
                    missing += 1
    finally:
        for store in stores.values():
            store.close()

    if missing:
        print(f"WARNING: Skipped {missing} missing caption(s) / database(s)")
    if not rows:
        raise ValueError(f"No captions found in {descriptions}. Check dataset and description database paths")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--grid", default="outputs/grids/world.aggrid")
    parser.add_argument("--dataset", default=f"{DATASET_PATH}/world")
    parser.add_argument("--train-dataset", default=None,
                        help="Required with --split external; verifies the datasets share no panorama IDs")
    parser.add_argument("--descriptions", default=None)
    parser.add_argument("--model", default=SIGLIP_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-panoramas", type=int, default=250,
                        help="Number of source panoramas sampled from the index (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-ids", default=None,
                        help="Optional newline-delimited panorama IDs for identical paired queries across galleries")
    parser.add_argument("--split", choices=("auto", "heldout", "in_sample", "external"), default="auto",
                        help="auto: prefer held-out and warn if only in-sample is possible")
    parser.add_argument("--prompts", type=int, nargs="+", choices=(1, 2, 3, 4), default=(1, 2, 3, 4))
    parser.add_argument("--aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--context-mix", type=float, default=1.0)
    parser.add_argument("--semantic-view-mode", choices=("all", "crops", "tokens"), default="all")
    parser.add_argument("--semantic-token-mix", type=float, default=None,
                        help="Blend separate crop and semantic-token retrieval scores (0=crops, 1=tokens)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (args.num_panoramas < 0 or args.seed < 0 or args.top_k < 1
            or not 0 <= args.context_mix <= 1
            or (args.semantic_token_mix is not None and not 0 <= args.semantic_token_mix <= 1)):
        parser.error("Counts and seed must be nonnegative, and --top-k must be positive")
    if args.split == "external" and not args.train_dataset:
        parser.error("--split external requires --train-dataset")

    import torch

    from modules.dataset import PanoramaDataset
    from modules.inference import GeoSearchInference, load_reference_index

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config") or {}
    del checkpoint

    index = load_reference_index(args.index)
    metadata = PanoramaDataset.load_metadata(args.dataset)
    external_train_ids = None
    if args.split == "external":
        if Path(args.dataset).resolve() == Path(args.train_dataset).resolve():
            parser.error("--dataset and --train-dataset must differ for external evaluation")
        external_train_ids = set(PanoramaDataset.load_metadata(args.train_dataset))
    records = index["records"]
    count = args.num_panoramas
    if args.positive_ids:
        positive_ids = [line.strip() for line in Path(args.positive_ids).read_text().splitlines()
                        if line.strip()]
        if not positive_ids or len(positive_ids) != len(set(positive_ids)):
            raise ValueError("--positive-ids must contain distinct, nonempty IDs")
        requested = set(positive_ids)
        records = [record for record in records if record["image_id"] in requested]
        if len(records) != len(requested):
            raise ValueError("Some requested paired positives are missing from the gallery index")
        count = 0  # Select exactly these positives, without resampling.
    selected, split = select_records(records, metadata, config,
                                     count, args.seed, args.split, external_train_ids)
    if split == "in_sample":
        print("WARNING: This checkpoint has no identifiable held-out set. Results may be in-sample.")
    descriptions = Path(args.descriptions) if args.descriptions else Path(args.dataset) / "descriptions"
    caption_rows = load_captions(selected, metadata, descriptions, args.prompts, args.seed)
    del index, metadata

    searcher = GeoSearchInference(
        checkpoint=args.checkpoint, index=args.index, grid=args.grid,
        model_name=args.model, device=args.device, aggregation=args.aggregation,
        context_mix=args.context_mix,
        semantic_view_mode=args.semantic_view_mode,
        semantic_token_mix=args.semantic_token_mix,
    )
    k = min(args.top_k, searcher.gallery_count)
    retrieved = searcher.retrieve_many([r["caption"] for r in caption_rows], top_k=k)

    results = []
    for row, result in zip(caption_rows, retrieved):
        rank = next((match["rank"] for match in result["matches"]
                     if match["image_id"] == row["image_id"]), 0)
        results.append({**row, "rank": rank, "truncated": result["truncated"],
                        "token_length": result["token_length"],
                        "top1_image_id": result["matches"][0]["image_id"]})

    summary = {
        "checkpoint": args.checkpoint, "index": args.index, "split": split,
        "num_panoramas": len(selected), "gallery_size": searcher.gallery_count,
        "aggregation": args.aggregation, "seed": args.seed, "top_k": k,
        "context_mix": args.context_mix,
        "semantic_view_mode": args.semantic_view_mode,
        "semantic_token_mix": args.semantic_token_mix,
        **summarize_rows(results, k),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    with (output / "queries.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Paired retrieval ({split}, {args.aggregation}): {len(results)} captions, "
          f"Recall@1={summary['overall']['recall@1']:.1%}, "
          f"Recall@10={summary['overall']['recall@10']:.1%}; saved to {output}")


if __name__ == "__main__":
    main()
