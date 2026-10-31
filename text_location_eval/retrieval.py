"""Retrieve ranked geographic panorama references for text-only queries."""

from __future__ import annotations

import random
from pathlib import Path

from .common import file_sha256, load_queries, write_json
from .indexing import expected_signature, load_index


def retrieve(args) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor

    from adaptive_geogrid import load_grid
    from modules.model import GeoSearchModel

    index = load_index(args.index)
    if index["signature"] != expected_signature(args):
        raise ValueError("Checkpoint, grid, model, or data differs from the index. Rebuild the index first.")

    queries = load_queries(args.queries)
    if args.limit_queries:
        queries = queries[:args.limit_queries]
    if args.top_k > len(index["records"]):
        raise ValueError("--top-k exceeds the number of indexed panoramas")

    model = GeoSearchModel(load_grid(args.grid), model_name=args.model).to(args.device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model_processor = AutoProcessor.from_pretrained(args.model)
    del checkpoint

    text_limit = model.backbone.config.text_config.max_position_embeddings
    texts = [row["query"] for row in queries]
    lengths = [len(tokens) for tokens in model_processor.tokenizer(
        texts, truncation=False, add_special_tokens=True)["input_ids"]]
    if any(length > text_limit for length in lengths):
        print(f"Warning: {sum(x > text_limit for x in lengths)} queries exceed the model's "
              f"{text_limit}-token limit and will be truncated. See retrieval output.")

    with torch.inference_mode():
        parts = []
        for start in range(0, len(texts), args.text_batch_size):
            inputs = model_processor(text=texts[start:start + args.text_batch_size],
                                     padding="max_length", truncation=True, return_tensors="pt")
            inputs = {k: v.to(args.device) for k, v in inputs.items()
                      if k in {"input_ids", "attention_mask"}}
            parts.append(model.encode_text(inputs).float())
        text_features = F.normalize(torch.cat(parts, dim=0), dim=-1)

        best_scores = torch.full((len(queries), args.top_k), -float("inf"), device=args.device)
        best_ids = torch.full((len(queries), args.top_k), -1, dtype=torch.long, device=args.device)
        for start in range(0, len(index["records"]), args.reference_chunk_size):
            crops = F.normalize(index["embeddings"][start:start + args.reference_chunk_size]
                                .float().to(args.device), dim=-1)
            if args.aggregation == "mean":
                panorama_features = F.normalize(crops.mean(dim=1), dim=-1)
                scores = text_features @ panorama_features.T
            else:
                scores = torch.einsum("qd,nvd->qnv", text_features, crops).amax(dim=-1)
            ids = torch.arange(start, start + crops.shape[0], device=args.device)
            ids = ids.unsqueeze(0).expand(len(queries), -1)
            merged_scores = torch.cat((best_scores, scores), dim=1)
            merged_ids = torch.cat((best_ids, ids), dim=1)
            best_scores, positions = merged_scores.topk(args.top_k, dim=1)
            best_ids = merged_ids.gather(1, positions)

    ranked = []
    for qidx, query in enumerate(queries):
        top_ids = best_ids[qidx].tolist()
        matches = [
            {"rank": rank, "similarity": float(score), **index["records"][item]}
            for rank, (score, item) in enumerate(zip(best_scores[qidx].tolist(), top_ids), 1)
        ]
        excluded = set(top_ids)
        others = [j for j in range(len(index["records"])) if j not in excluded]
        rng = random.Random(args.seed + qidx * 1000003)
        controls = [index["records"][j] for j in rng.sample(
            others, k=min(args.controls, len(others)))]
        ranked.append({**query, "truncated": lengths[qidx] > text_limit,
                       "token_length": lengths[qidx], "matches": matches,
                       "random_controls": controls})

    write_json(args.output, {
        "index_signature": index["signature"],
        "index_count": len(index["records"]),
        "queries_sha256": file_sha256(args.queries),
        "aggregation": args.aggregation,
        "top_k": args.top_k,
        "queries": ranked,
    })
    print(f"Saved retrieval for {len(ranked)} text queries to {args.output}")
