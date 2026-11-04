#!/bin/bash
# Called by the training scripts after training finishes.
set -euo pipefail

CHECKPOINT="$1"
GRID="outputs/grids/world.aggrid"
NUM_IMAGES=20000
CHECKPOINT_NAME="${CHECKPOINT##*/}"
CHECKPOINT_STEM="${CHECKPOINT_NAME%.*}"
CHECKPOINT_HASH="$(sha256sum -- "$CHECKPOINT" | cut -c1-12)"
INDEX="outputs/indexes/${CHECKPOINT_STEM}-${CHECKPOINT_HASH}-${NUM_IMAGES}.pt"
RUN_DIR="outputs/eval/text_location_eval/${CHECKPOINT_STEM}-${CHECKPOINT_HASH}"

python -m text_location_eval index \
    --checkpoint "$CHECKPOINT" --grid "$GRID" \
    --num-images "$NUM_IMAGES" --batch-size 8 --workers 4 \
    --save-every 20000 --output "$INDEX"

python -m text_location_eval retrieve \
    --checkpoint "$CHECKPOINT" --grid "$GRID" \
    --num-images "$NUM_IMAGES" --index "$INDEX" \
    --output "$RUN_DIR/retrieval.json"

RETRIEVAL_HASH="$(sha256sum -- "$RUN_DIR/retrieval.json" | cut -c1-12)"
JUDGE_DIR="$RUN_DIR/judge-$RETRIEVAL_HASH"

python -m text_location_eval judge \
    --retrieval "$RUN_DIR/retrieval.json" --output "$JUDGE_DIR"

python -m text_location_eval metrics \
    --retrieval "$RUN_DIR/retrieval.json" \
    --analysis "$JUDGE_DIR/query_analysis.jsonl" \
    --judgments "$JUDGE_DIR/judgments.jsonl" --output "$JUDGE_DIR"

echo "Evaluation results: $JUDGE_DIR"
