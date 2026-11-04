"""Run as: python -m text_location_eval {index,retrieve,judge,metrics}."""

from __future__ import annotations

import argparse

from config import DATASET_PATH, MODEL, SIGLIP_MODEL


RUN = "outputs/eval/text_location_eval/dev"
QUERIES = "text_location_eval/deepstreetsearch_text_queries_v1.jsonl"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent text-to-location evaluation")
    commands = parser.add_subparsers(dest="command", required=True)

    def model_args(p):
        p.add_argument("--checkpoint", default="checkpoints/deepstreetsearchFULL.pt")
        p.add_argument("--grid", default="outputs/grids/world.aggrid")
        p.add_argument("--dataset", default=f"{DATASET_PATH}/world")
        p.add_argument("--model", default=SIGLIP_MODEL)
        p.add_argument("--num-images", type=int, default=5000)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", default="cuda")

    index = commands.add_parser("index", help="Sample training panoramas and cache their crop embeddings")
    model_args(index)
    index.add_argument("--batch-size", type=int, default=16)
    index.add_argument("--workers", type=int, default=8)
    index.add_argument("--save-every", type=int, default=5000,
                       help="Save resumable embedding chunks every N panoramas")
    index.add_argument("--output", default=f"{RUN}/reference.pt")
    index.add_argument("--rebuild", action="store_true")

    retrieve = commands.add_parser("retrieve", help="Retrieve top-K locations for each text query")
    model_args(retrieve)
    retrieve.add_argument("--index", default=f"{RUN}/reference.pt")
    retrieve.add_argument("--queries", default=QUERIES)
    retrieve.add_argument("--output", default=f"{RUN}/retrieval.json")
    retrieve.add_argument("--top-k", type=int, default=10)
    retrieve.add_argument("--controls", type=int, default=5)
    retrieve.add_argument("--gallery-size", type=int, default=0,
                          help="Search only N indexed panoramas (0 = full index)")
    retrieve.add_argument("--gallery-seed", type=int, default=42,
                          help="Seed for deterministic, nested gallery subsets")
    retrieve.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    retrieve.add_argument("--reference-chunk-size", type=int, default=2048)
    retrieve.add_argument("--text-batch-size", type=int, default=16)
    retrieve.add_argument("--limit-queries", type=int, default=0,
                          help="For smoke tests only (0 evaluates all prompts)")

    judge = commands.add_parser("judge", help="Run blinded Qwen3-VL category and visual judgments")
    judge.add_argument("--retrieval", default=f"{RUN}/retrieval.json")
    judge.add_argument("--output", default=RUN)
    judge.add_argument("--judge-model", default=MODEL)
    judge.add_argument("--judge-batch-size", type=int, default=4)
    judge.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    judge.add_argument("--max-model-len", type=int, default=4096)
    judge.add_argument("--max-tokens", type=int, default=650)

    metrics = commands.add_parser("metrics", help="Calculate nDCG, precision, and category metrics")
    metrics.add_argument("--retrieval", default=f"{RUN}/retrieval.json")
    metrics.add_argument("--analysis", default=f"{RUN}/query_analysis.jsonl")
    metrics.add_argument("--judgments", default=f"{RUN}/judgments.jsonl")
    metrics.add_argument("--output", default=RUN)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.command in {"index", "retrieve"}:
        if args.num_images < 1 or args.seed < 0:
            parser.error("--num-images must be positive and --seed nonnegative")
    if args.command == "index":
        if args.batch_size < 1 or args.workers < 0 or args.save_every < 1:
            parser.error("--batch-size/--save-every must be positive and --workers nonnegative")
        from .indexing import build_index
        build_index(args)
    elif args.command == "retrieve":
        if min(args.top_k, args.text_batch_size, args.reference_chunk_size) < 1 \
                or args.controls < 0 or args.limit_queries < 0 \
                or args.gallery_size < 0 or args.gallery_seed < 0:
            parser.error("Invalid retrieval count/batch size")
        from .retrieval import retrieve
        retrieve(args)
    elif args.command == "judge":
        if args.judge_batch_size < 1 or args.max_model_len < 1 or args.max_tokens < 1 \
                or not 0 < args.gpu_memory_utilization <= 1:
            parser.error("Invalid judge settings")
        from .judging import judge_retrieval
        judge_retrieval(args)
    else:
        from .metrics import summarize
        summarize(args)


if __name__ == "__main__":
    main()
