#!/bin/bash
#SBATCH --job-name=LaVPR-Eval
#SBATCH --partition=GPU24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%A.out
#SBATCH --error=R-%x.%A.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source .venv/bin/activate

# Pass the SAME source index, gallery size, and seed used for DeepStreetSearch.
REFERENCE_INDEX="${1:-outputs/indexes/base_index.pt}"
GALLERY_SIZE="${2:-20000}"
GALLERY_SEED="${3:-42}"
AGGREGATION="${4:-max}"
QUERIES="text_location_eval/deepstreetsearch_text_queries_v1.jsonl"

SOURCE_NAME="$(basename "$REFERENCE_INDEX" .pt)"
SOURCE_HASH="$(sha256sum "$REFERENCE_INDEX" | cut -c1-12)"
INDEX="outputs/indexes/lavpr/${SOURCE_NAME}-${SOURCE_HASH}-gallery-${GALLERY_SIZE}-seed-${GALLERY_SEED}.pt"
RUN_DIR="outputs/eval/lavpr/${SOURCE_NAME}-${SOURCE_HASH}/gallery-${GALLERY_SIZE}-seed-${GALLERY_SEED}/${AGGREGATION}"

mkdir -p "$(dirname "$INDEX")" "$RUN_DIR"

python -m external_eval.lavpr index \
  --reference-index "$REFERENCE_INDEX" \
  --gallery-size "$GALLERY_SIZE" \
  --gallery-seed "$GALLERY_SEED" \
  --batch-size 4 \
  --save-every 256 \
  --output "$INDEX"

python -m external_eval.lavpr retrieve \
  --index "$INDEX" \
  --queries "$QUERIES" \
  --aggregation "$AGGREGATION" \
  --top-k 10 \
  --controls 5 \
  --output "$RUN_DIR/retrieval.json"

# Exactly the existing independent Qwen judging and metric calculation.
RETRIEVAL_HASH="$(sha256sum "$RUN_DIR/retrieval.json" | cut -c1-12)"
JUDGE_DIR="$RUN_DIR/judge-$RETRIEVAL_HASH"
mkdir -p "$JUDGE_DIR"

python -m text_location_eval judge \
  --retrieval "$RUN_DIR/retrieval.json" \
  --output "$JUDGE_DIR"

python -m text_location_eval metrics \
  --retrieval "$RUN_DIR/retrieval.json" \
  --analysis "$JUDGE_DIR/query_analysis.jsonl" \
  --judgments "$JUDGE_DIR/judgments.jsonl" \
  --output "$JUDGE_DIR"

echo "LaVPR evaluation: $JUDGE_DIR"
