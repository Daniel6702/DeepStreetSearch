"""Retrieval statistics computed from independently stored relevance judgments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .common import CATEGORIES, load_jsonl, write_json


def dcg(grades: list[int]) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 2)
               for rank, grade in enumerate(grades))


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """nDCG within the judged pool; NOT full-gallery nDCG."""
    if k <= 0:
        raise ValueError("k must be positive")
    missing = set(ranked_ids[:k]) - relevance.keys()
    if missing:
        raise ValueError(f"Missing relevance judgments for {sorted(missing)[:3]}")
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return dcg([relevance[item] for item in ranked_ids[:k]]) / ideal if ideal else 0.0


def write_csv(path: Path, records: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def calculate_metrics(retrieval: dict, analyses: dict, judgments: dict) -> dict:
    per_query = []
    skipped_queries = []
    per_category = defaultdict(list)
    expected_pairs = 0
    completed_pairs = 0
    for query in retrieval["queries"]:
        qid = query["id"]
        matches = query["matches"]
        candidates = [*matches, *query.get("random_controls", [])]
        expected_pairs += len(candidates)
        completed_pairs += sum((qid, item["image_id"]) in judgments for item in candidates)
        if qid not in analyses:
            skipped_queries.append({"query_id": qid, "reason": "missing_query_analysis"})
            continue
        active = analyses[qid]["categories"]
        if not matches:
            raise ValueError(f"No retrieval matches for {qid}")
        missing = [item["image_id"] for item in candidates
                   if (qid, item["image_id"]) not in judgments]
        if missing:
            skipped_queries.append({
                "query_id": qid, "reason": "missing_visual_judgments",
                "missing_image_ids": missing,
            })
            continue
        judged_pool = {}
        for item in candidates:
            key = (qid, item["image_id"])
            judged_pool[item["image_id"]] = judgments[key]["holistic_score"]

        ranks = [item["image_id"] for item in matches]
        grades = [judged_pool[item] for item in ranks]
        weighted_category = []
        for item in matches:
            judgment = judgments[(qid, item["image_id"])]
            weight_sum = 0
            weighted_sum = 0
            for name, definition in active.items():
                if name not in CATEGORIES:
                    raise ValueError(f"Invalid active category: {name}")
                value = judgment["category_scores"][name]["score"]
                importance = definition["importance"]
                per_category[name].append(value)
                weighted_sum += value * importance
                weight_sum += importance
            if not weight_sum:
                raise ValueError(f"Query {qid} has no applicable categories")
            weighted_category.append(weighted_sum / weight_sum)

        top5 = grades[:5]
        top10 = grades[:10]
        control_scores = [judged_pool[item["image_id"]]
                          for item in query.get("random_controls", [])]
        per_query.append({
            "query_id": qid,
            "theme": query.get("theme", ""),
            "geo_context": query.get("geo_context", ""),
            "truncated": query.get("truncated", False),
            "nDCG@10": ndcg_at_k(ranks, judged_pool, min(10, len(ranks))),
            "P@5": sum(s >= 2 for s in top5) / len(top5),
            "P@10": sum(s >= 2 for s in top10) / len(top10),
            "Success@10": float(any(s >= 2 for s in top10)),
            "mean_relevance@5": sum(top5) / len(top5),
            "mean_relevance@10": sum(top10) / len(top10),
            "top1_relevance": grades[0],
            "mean_weighted_categories@10": sum(weighted_category[:10]) / len(top10),
            "random_control_mean": sum(control_scores) / len(control_scores)
            if control_scores else None,
        })

    numeric = ["nDCG@10", "P@5", "P@10", "Success@10", "mean_relevance@5",
               "mean_relevance@10", "top1_relevance", "mean_weighted_categories@10",
               "random_control_mean"]
    aggregate = {
        metric: sum(row[metric] for row in per_query if row[metric] is not None) /
        sum(row[metric] is not None for row in per_query)
        if any(row[metric] is not None for row in per_query) else None
        for metric in numeric
    }
    categories = [
        {"category": name, "mean_score": sum(values) / len(values), "judgments": len(values)}
        for name, values in sorted(per_category.items())
    ]
    return {
        "summary": {"num_queries": len(per_query),
                    "num_queries_total": len(retrieval["queries"]),
                    "num_queries_skipped": len(skipped_queries),
                    "status": "complete" if not skipped_queries else "partial",
                    "judgments_completed": completed_pairs,
                    "judgments_required": expected_pairs,
                    "judgment_coverage": completed_pairs / expected_pairs if expected_pairs else 0.0,
                    "index_count": retrieval["index_count"],
                    "aggregation": retrieval["aggregation"],
                    "queries_truncated": sum(q["truncated"] for q in per_query), **aggregate},
        "per_query": per_query,
        "skipped_queries": skipped_queries,
        "per_category": categories,
        "limitations": [
            "Incomplete queries are excluded from metric averages, never treated as zero relevance.",
            "Development data: the reference gallery is sampled from the training set.",
            "The prompts are synthetic and the judgments are produced by a VLM, not ground truth.",
            "nDCG@10 is normalized against the judged candidate pool only (retrievals plus random controls).",
            "nDCG measures ranking, not absolute quality; report P@K and category scores alongside it.",
            "Explicit geographic names are not geocoded or independently verified in this stage.",
            "A missing relevant image in the gallery cannot be distinguished from unsuccessful retrieval.",
        ],
    }


def summarize(args) -> None:
    with Path(args.retrieval).open("r", encoding="utf-8") as stream:
        retrieval = json.load(stream)
    analyses = {row["query_id"]: row for row in load_jsonl(args.analysis)} \
        if Path(args.analysis).exists() else {}
    judgments = {(row["query_id"], row["image_id"]): row for row in load_jsonl(args.judgments)} \
        if Path(args.judgments).exists() else {}
    result = calculate_metrics(retrieval, analyses, judgments)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", result)
    query_columns = [
        "query_id", "theme", "geo_context", "truncated", "nDCG@10",
        "P@5", "P@10", "Success@10", "mean_relevance@5",
        "mean_relevance@10", "top1_relevance",
        "mean_weighted_categories@10", "random_control_mean",
    ]
    write_csv(output / "per_query.csv", result["per_query"], query_columns)
    write_csv(output / "per_category.csv", result["per_category"],
              ["category", "mean_score", "judgments"])
    print(json.dumps(result["summary"], indent=2))
    if result["skipped_queries"]:
        print(f"WARNING: {len(result['skipped_queries'])} incomplete queries excluded from "
              "metric averages. See metrics.json / skipped_queries; rerun judge to retry.")
    print(f"Saved metrics in {output}")
