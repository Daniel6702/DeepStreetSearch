"""Evaluate text retrieval using the same inference path as applications."""

from __future__ import annotations

import random

from .common import file_sha256, load_queries, write_json
from .indexing import expected_signature


def random_gallery_controls(searcher, matches: list[dict], count: int,
                            seed: int) -> list[dict]:
    """Sample controls only from the active gallery, excluding top-K matches.

    Rejection sampling avoids rebuilding a million-item pool for every query;
    the small-gallery fallback handles cases where nearly all items are matched.
    """
    if count <= 0:
        return []
    excluded = {item["image_id"] for item in matches}
    available = searcher.gallery_count - len(excluded)
    target = min(count, available)
    if target <= 0:
        return []

    rng = random.Random(seed)
    if target * 3 >= available:
        candidates = [i for i in searcher.gallery_indices
                      if searcher.records[i]["image_id"] not in excluded]
        return [searcher.records[i] for i in rng.sample(candidates, k=target)]

    chosen = set()
    controls = []
    while len(controls) < target:
        pos = rng.randrange(searcher.gallery_count)
        index_pos = searcher.gallery_indices[pos]
        record = searcher.records[index_pos]
        if index_pos not in chosen and record["image_id"] not in excluded:
            chosen.add(index_pos)
            controls.append(record)
    return controls


def retrieve(args) -> None:
    from modules.inference import GeoSearchInference

    # Evaluation still requires the full signature, including the training
    # dataset sample. gallery_size only changes which entries are searched.
    gallery_size = getattr(args, "gallery_size", 0) or None
    gallery_seed = getattr(args, "gallery_seed", 42)
    searcher = GeoSearchInference(
        checkpoint=args.checkpoint, index=args.index, grid=args.grid,
        model_name=args.model, device=args.device, gallery_size=gallery_size,
        gallery_seed=gallery_seed, aggregation=args.aggregation,
        reference_chunk_size=args.reference_chunk_size,
        text_batch_size=args.text_batch_size,
        expected_signature=expected_signature(args),
    )

    queries = load_queries(args.queries)
    if args.limit_queries:
        queries = queries[:args.limit_queries]
    results = searcher.retrieve_many([row["query"] for row in queries], top_k=args.top_k)

    ranked = []
    for qidx, (query, result) in enumerate(zip(queries, results)):
        controls = random_gallery_controls(
            searcher, result["matches"], args.controls, args.seed + qidx * 1000003)
        ranked.append({**query, "truncated": result["truncated"],
                       "token_length": result["token_length"],
                       "matches": result["matches"], "random_controls": controls})

    write_json(args.output, {
        "index_signature": searcher.index["signature"],
        "index_count": searcher.gallery_count,
        "index_total_count": len(searcher.records),
        "gallery_seed": gallery_seed,
        "queries_sha256": file_sha256(args.queries),
        "aggregation": args.aggregation,
        "top_k": args.top_k,
        "queries": ranked,
    })
    print(f"Saved retrieval for {len(ranked)} text queries against "
          f"{searcher.gallery_count:,}/{len(searcher.records):,} panoramas to {args.output}")
