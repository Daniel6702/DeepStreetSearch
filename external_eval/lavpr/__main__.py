"""LaVPR on the exact DeepStreetSearch gallery and text relevance benchmark.

Usage:
  python -m external_eval.lavpr index --reference-index ... --output ...
  python -m external_eval.lavpr retrieve --index ... --output ...

The existing text_location_eval retrieve/judge/metrics implementation is unchanged.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from .model import BASE_MODEL


DEFAULT_LAVPR = "external/LaVPR"
DEFAULT_QUERIES = "text_location_eval/deepstreetsearch_text_queries_v1.jsonl"


def _signature(args, source_index: dict) -> dict:
    from text_location_eval.common import file_sha256

    adapter = Path(args.adapter).resolve()
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"LaVPR LoRA adapter not found: {weights}")
    return {
        "method": "LaVPR-cross-modal-blip-six-views-v1",
        "base_model": BASE_MODEL,
        "adapter_sha256": file_sha256(weights),
        "adapter_code_sha256": file_sha256(Path(__file__).with_name("model.py")),
        "reference_sha256": file_sha256(args.reference_index),
        "reference_signature": source_index["signature"],
        "gallery_size": args.gallery_size,
        "gallery_seed": args.gallery_seed,
        "image_size": 384,
    }


def _write_part(torch, directory: Path, number: int, signature: dict,
                ids: list[str], vectors: list) -> None:
    """Save finished features atomically, allowing interrupted indexing to resume."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"part-{number:06d}.pt"
    temporary = destination.with_suffix(".tmp")
    torch.save({"signature": signature, "ids": ids,
                "embeddings": torch.stack(vectors).half()}, temporary)
    temporary.replace(destination)


def _load_parts(torch, directory: Path, signature: dict) -> tuple[dict, int]:
    cached = {}
    paths = sorted(directory.glob("part-*.pt")) if directory.is_dir() else []
    for number, path in enumerate(paths):
        if path.name != f"part-{number:06d}.pt":
            raise ValueError(f"Unexpected partial-index sequence in {directory}")
        part = torch.load(path, map_location="cpu", weights_only=True)
        ids, embeddings = part.get("ids"), part.get("embeddings")
        if (part.get("signature") != signature or not isinstance(ids, list)
                or embeddings is None or embeddings.ndim != 3 or embeddings.shape[1:] != (6, 256)
                or len(ids) != len(embeddings)):
            raise ValueError(f"Incompatible LaVPR partial index: {path}")
        for image_id, embedding in zip(ids, embeddings):
            if image_id in cached:
                raise ValueError(f"Duplicate image in LaVPR partial index: {image_id}")
            cached[image_id] = embedding
    return cached, len(paths)


def index_gallery(args) -> None:
    import torch
    from tqdm import tqdm

    from modules.dataset.shards import image_key, iter_shard_images
    from modules.inference import load_reference_index, select_gallery_indices
    from modules.panorama import split_panorama
    from .model import encode_images, load_model

    source = load_reference_index(args.reference_index)
    all_records = source["records"]
    indices = select_gallery_indices(
        len(all_records), args.gallery_size or None, args.gallery_seed,
        image_ids=[row["image_id"] for row in all_records],
    )
    records = [all_records[i] for i in indices]
    if len({row["image_id"] for row in records}) != len(records):
        raise ValueError("Duplicate panorama identifiers in source gallery")
    args.gallery_size = len(records)
    signature = _signature(args, source)
    output = Path(args.output)
    if output.is_file():
        existing = torch.load(output, map_location="cpu", weights_only=True)
        if (existing.get("signature") != signature
                or [row["image_id"] for row in existing["records"]]
                != [row["image_id"] for row in records]
                or existing["embeddings"].shape != (len(records), 6, 256)):
            raise ValueError(f"Existing index belongs to another gallery/model: {output}")
        print(f"Existing LaVPR index valid: {len(records):,} panoramas in {output}")
        return

    parts_dir = output.with_name(output.stem + ".parts")
    completed, part_number = _load_parts(torch, parts_dir, signature)
    wanted = {row["image_id"] for row in records}
    if not set(completed).issubset(wanted):
        raise ValueError("Saved LaVPR parts contain images outside this gallery")

    by_shard = defaultdict(set)
    for row in records:
        if row["image_id"] not in completed:
            by_shard[row["shard"]].add(row["image_id"])

    new_ids, new_vectors = [], []
    pending_ids, pending_crops = [], []
    model = transform = None
    if by_shard:
        model, _, transform = load_model(args.lavpr, args.adapter, args.device)

    def flush_batch():
        nonlocal part_number
        if not pending_ids:
            return
        try:
            vectors = encode_images(model, transform, pending_crops, args.device)
            if vectors.shape != (len(pending_ids) * 6, 256):
                raise ValueError(f"Unexpected LaVPR image embedding shape: {vectors.shape}")
            for image_id, vector in zip(pending_ids, vectors.reshape(-1, 6, 256)):
                new_ids.append(image_id)
                new_vectors.append(vector)
        finally:
            for crop in pending_crops:
                crop.close()
            pending_ids.clear()
            pending_crops.clear()
        if len(new_ids) >= args.save_every:
            _write_part(torch, parts_dir, part_number, signature, new_ids, new_vectors)
            completed.update(zip(new_ids, new_vectors))
            part_number += 1
            print(f"Saved LaVPR features: {len(completed):,}/{len(records):,}")
            new_ids.clear()
            new_vectors.clear()

    for shard, selected in tqdm(sorted(by_shard.items()), desc="LaVPR panorama shards"):
        if not Path(shard).is_file():
            raise FileNotFoundError(f"Original panorama shard unavailable: {shard}")
        found = set()
        for filename, panorama in iter_shard_images(shard, allowed_keys=selected):
            image_id = image_key(filename)
            crops = []
            try:
                crops = split_panorama(panorama)
                if len(crops) != 6:
                    raise ValueError(f"Expected six views for panorama {image_id}")
                pending_crops.extend(crop.image for crop in crops)
                pending_ids.append(image_id)
                found.add(image_id)
                crops = []  # Ownership of crop images transferred to pending_crops.
                if len(pending_ids) >= args.batch_size:
                    flush_batch()
            finally:
                panorama.close()
                for crop in crops:
                    crop.image.close()
        if selected != found:
            raise FileNotFoundError(
                f"Shard {shard} missing {len(selected - found)} requested panoramas: "
                f"{sorted(selected - found)[:3]}")

    flush_batch()
    if new_ids:
        _write_part(torch, parts_dir, part_number, signature, new_ids, new_vectors)
        completed.update(zip(new_ids, new_vectors))

    if set(completed) != wanted:
        raise ValueError(f"Indexed {len(completed):,}/{len(records):,} requested panoramas")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save({"signature": signature, "records": records,
                "embeddings": torch.stack([completed[row["image_id"]]
                                           for row in records]).half()}, temporary)
    temporary.replace(output)
    print(f"Saved LaVPR gallery: {len(records):,} panoramas to {output}")


def rank_queries(query_vectors, gallery_vectors, top_k: int, aggregation: str,
                 chunk_size: int, device: str):
    """Return descending (cosine score, panorama position) for each query."""
    import torch
    import torch.nn.functional as F

    if gallery_vectors.ndim != 3 or gallery_vectors.shape[1:] != (6, 256):
        raise ValueError("LaVPR image index must have shape [panoramas, 6, 256]")
    if not 1 <= top_k <= len(gallery_vectors):
        raise ValueError("top_k must fit within the gallery")
    queries = F.normalize(query_vectors.to(device).float(), dim=-1)
    best_scores = queries.new_empty((len(queries), 0))
    best_indices = torch.empty((len(queries), 0), dtype=torch.long, device=device)

    for start in range(0, len(gallery_vectors), chunk_size):
        views = gallery_vectors[start:start + chunk_size].to(device).float()
        views = F.normalize(views, dim=-1)
        if aggregation == "mean":
            scores = queries @ F.normalize(views.mean(dim=1), dim=-1).T
        elif aggregation == "max":
            scores = (queries @ views.flatten(0, 1).T).reshape(len(queries), -1, 6).max(dim=-1).values
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        ids = torch.arange(start, start + len(views), device=device).expand(len(queries), -1)
        candidate_scores = torch.cat((best_scores, scores), dim=1)
        candidate_indices = torch.cat((best_indices, ids), dim=1)
        best_scores, positions = candidate_scores.topk(min(top_k, candidate_scores.shape[1]), dim=1)
        best_indices = candidate_indices.gather(1, positions)
    return best_scores.cpu().tolist(), best_indices.cpu().tolist()


def retrieve(args) -> None:
    import torch

    from text_location_eval.common import file_sha256, load_queries, write_json
    from text_location_eval.retrieval import random_gallery_controls
    from .model import encode_texts, load_model

    index = torch.load(args.index, map_location="cpu", weights_only=True)
    signature, records, gallery = index["signature"], index["records"], index["embeddings"]
    if signature.get("method") != "LaVPR-cross-modal-blip-six-views-v1":
        raise ValueError("Expected an index built by this LaVPR adapter")
    if (file_sha256(Path(args.adapter) / "adapter_model.safetensors")
            != signature["adapter_sha256"]):
        raise ValueError("The LaVPR LoRA weights differ from those used to build the index")
    if file_sha256(Path(__file__).with_name("model.py")) != signature["adapter_code_sha256"]:
        raise ValueError("The LaVPR encoder implementation changed; rebuild the index")
    if len(records) != len(gallery):
        raise ValueError("LaVPR index records and image embeddings differ in length")

    queries = load_queries(args.queries)
    if args.limit_queries:
        queries = queries[:args.limit_queries]
    model, processor, _ = load_model(args.lavpr, args.adapter, args.device)
    features, lengths = [], []
    for start in range(0, len(queries), args.text_batch_size):
        batch = [row["query"] for row in queries[start:start + args.text_batch_size]]
        embeddings, batch_lengths = encode_texts(model, processor, batch, args.device)
        features.append(embeddings)
        lengths.extend(batch_lengths)
    del model

    scores, ids = rank_queries(torch.cat(features), gallery, args.top_k,
                               args.aggregation, args.reference_chunk_size, args.device)
    searcher = SimpleNamespace(gallery_count=len(records), gallery_indices=range(len(records)),
                               records=records)
    ranked = []
    for i, row in enumerate(queries):
        matches = [{**records[index], "similarity": float(score)}
                   for score, index in zip(scores[i], ids[i])]
        controls = random_gallery_controls(searcher, matches, args.controls, args.seed + i * 1000003)
        ranked.append({**row, "truncated": False, "token_length": int(lengths[i]),
                       "matches": matches, "random_controls": controls})

    write_json(args.output, {
        "index_signature": signature,
        "index_count": len(records),
        "index_total_count": len(records),
        "gallery_seed": signature["gallery_seed"],
        "queries_sha256": file_sha256(args.queries),
        "aggregation": args.aggregation,
        "top_k": args.top_k,
        "queries": ranked,
    })
    print(f"Saved LaVPR retrieval for {len(queries)} queries against "
          f"{len(records):,} panoramas to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LaVPR on the existing Street View gallery")
    commands = parser.add_subparsers(dest="command", required=True)

    def model_args(p):
        p.add_argument("--lavpr", default=DEFAULT_LAVPR, help="Original LaVPR submodule")
        p.add_argument("--adapter", default=f"{DEFAULT_LAVPR}/checkpoints/blip_lora_all_r64")
        p.add_argument("--device", default="cuda")

    build = commands.add_parser("index", help="Encode the exact DeepStreetSearch gallery with LaVPR")
    model_args(build)
    build.add_argument("--reference-index", required=True,
                       help="Existing DeepStreetSearch .pt index identifying the panoramas")
    build.add_argument("--gallery-size", type=int, default=20000,
                       help="0 uses the entire supplied reference index")
    build.add_argument("--gallery-seed", type=int, default=42)
    build.add_argument("--batch-size", type=int, default=4, help="Panoramas per BLIP image batch")
    build.add_argument("--save-every", type=int, default=256, help="Panoramas per resumable part")
    build.add_argument("--output", default="outputs/indexes/lavpr_index.pt")

    search = commands.add_parser("retrieve", help="Encode text and write benchmark-compatible retrieval.json")
    model_args(search)
    search.add_argument("--index", default="outputs/indexes/lavpr_index.pt")
    search.add_argument("--queries", default=DEFAULT_QUERIES)
    search.add_argument("--output", default="outputs/eval/lavpr/retrieval.json")
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--controls", type=int, default=5)
    search.add_argument("--aggregation", choices=("mean", "max"), default="max")
    search.add_argument("--seed", type=int, default=42)
    search.add_argument("--text-batch-size", type=int, default=16)
    search.add_argument("--reference-chunk-size", type=int, default=2048)
    search.add_argument("--limit-queries", type=int, default=0)

    args = parser.parse_args()
    if args.command == "index":
        if min(args.batch_size, args.save_every) < 1 or args.gallery_size < 0 or args.gallery_seed < 0:
            parser.error("Invalid gallery settings")
        index_gallery(args)
    else:
        if (min(args.top_k, args.text_batch_size, args.reference_chunk_size) < 1
                or args.controls < 0 or args.seed < 0 or args.limit_queries < 0):
            parser.error("Invalid retrieval settings")
        retrieve(args)


if __name__ == "__main__":
    main()
