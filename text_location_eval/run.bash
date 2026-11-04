#!/bin/bash
#SBATCH --job-name=TextLocationEval
#SBATCH --partition=GPU48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%A.out
#SBATCH --error=R-%x.%A.err

set -euo pipefail

echo "Begin Setup"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate

CHECKPOINT="${1:-outputs/checkpoints/deepstreetsearch_E6.pt}"
NUM_IMAGES="${2:-20000}"
GALLERY_SIZE="${3:-20000}"
GALLERY_SEED="${4:-42}"

CHECKPOINT_NAME="${CHECKPOINT##*/}"
CHECKPOINT_STEM="${CHECKPOINT_NAME%.*}"
CHECKPOINT_HASH="$(sha256sum -- "$CHECKPOINT" | cut -c1-12)"

INDEX_DIR="outputs/indexes"
INDEX="${INDEX_DIR}/${CHECKPOINT_STEM}_index.pt"

RUN_DIR="outputs/eval/text_location_eval/${CHECKPOINT_STEM}-${CHECKPOINT_HASH}-${NUM_IMAGES}/gallery-${GALLERY_SIZE}-seed-${GALLERY_SEED}"

GRID="outputs/grids/world.aggrid"

mkdir -p "$RUN_DIR"

echo "Checkpoint: $CHECKPOINT"
echo "Index: $INDEX"
echo "Gallery size: $GALLERY_SIZE"
echo "Gallery seed: $GALLERY_SEED"
echo "Results: $RUN_DIR"

echo "Begin Eval"

# Create index of image embeddings (only needed once per checkpoint)
python -m text_location_eval index \
  --checkpoint "$CHECKPOINT" \
  --grid "$GRID" \
  --num-images "$NUM_IMAGES" \
  --batch-size 8 \
  --workers 4 \
  --output "$INDEX" \
  --save-every 20000

# Verify index exists
if [[ ! -f "$INDEX" ]]; then
    echo "ERROR: Index not found: $INDEX"
    exit 1
fi

# Retrieve top 10 locations for each test prompt
if [[ -f "$RUN_DIR/retrieval.json" ]]; then
    echo "Existing retrieval found. Skipping retrieval."
else
    python -m text_location_eval retrieve \
      --checkpoint "$CHECKPOINT" \
      --grid "$GRID" \
      --num-images "$NUM_IMAGES" \
      --index "$INDEX" \
      --gallery-size "$GALLERY_SIZE" \
      --gallery-seed "$GALLERY_SEED" \
      --output "$RUN_DIR/retrieval.json"
fi

# Use a separate judgment directory for each retrieval
RETRIEVAL_HASH="$(sha256sum "$RUN_DIR/retrieval.json" | cut -c1-12)"
JUDGE_DIR="${RUN_DIR}/judge-${RETRIEVAL_HASH}"

mkdir -p "$JUDGE_DIR"

echo "Judge results: $JUDGE_DIR"

# Evaluate retrieved locations using Qwen (resumes existing judgments)
python -m text_location_eval judge \
  --retrieval "$RUN_DIR/retrieval.json" \
  --output "$JUDGE_DIR"

# Calculate evaluation metrics
python -m text_location_eval metrics \
  --retrieval "$RUN_DIR/retrieval.json" \
  --analysis "$JUDGE_DIR/query_analysis.jsonl" \
  --judgments "$JUDGE_DIR/judgments.jsonl" \
  --output "$JUDGE_DIR"

echo "Eval Finished"