#!/bin/bash
#SBATCH --job-name=DeepStreetSearch
#SBATCH --partition=GPU24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
# SBATCH --nodelist=node4
#SBATCH --output=R-%x.%A.out
#SBATCH --error=R-%x.%A.err

set -euo pipefail

cd ~/DeepStreetSearch
source .venv/bin/activate

date
nvidia-smi

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

GRID="${GRID:-outputs/grids/world.aggrid}"
LABELS="${LABELS:-outputs/grids/world_geo_labels.npz}"

# 1 = exclude other crops from the same panorama as contrastive negatives.
# 0 = keep the original contrastive loss.
REVISED_CONTRASTIVE_POSITIVES="${REVISED_CONTRASTIVE_POSITIVES:-1}"
CHECKPOINT="outputs/checkpoints/checkpoint_woGEOL_5P_E8.pt"

echo "Training checkpoint: $CHECKPOINT"
echo "Revised contrastive positives: $REVISED_CONTRASTIVE_POSITIVES"

python -u train.py \
    --grid "$GRID" \
    --geo-label-cache "$LABELS" \
    --dataset-fraction 0.05 \
    --seed 42 \
    --epochs 8 \
    --batch-size 4 \
    --workers 8 \
    --description-mode random \
    --learning-rate 1e-5 \
    --lr-scheduler cosine \
    --warmup-ratio 0.02 \
    --max-grad-norm 1.0 \
    --geo-loss-weight 0.0 \
    --checkpoint "$CHECKPOINT" \
#--revised-contrastive-positives

# Run the independent text-to-location evaluation after successful training.
# Use EVAL_NUM_IMAGES=1000000 for a larger index, or EVAL_ENABLED=0 to skip.
EVAL_ENABLED="${EVAL_ENABLED:-1}"
if [[ "$EVAL_ENABLED" == "1" ]]; then
    EVAL_NUM_IMAGES="${EVAL_NUM_IMAGES:-20000}"
    EVAL_GALLERY_SIZE="${EVAL_GALLERY_SIZE:-$EVAL_NUM_IMAGES}"
    EVAL_GALLERY_SEED="${EVAL_GALLERY_SEED:-42}"

    CHECKPOINT_NAME="${CHECKPOINT##*/}"
    CHECKPOINT_STEM="${CHECKPOINT_NAME%.*}"
    CHECKPOINT_HASH="$(sha256sum -- "$CHECKPOINT" | cut -c1-12)"
    INDEX="outputs/indexes/${CHECKPOINT_STEM}-${CHECKPOINT_HASH}-${EVAL_NUM_IMAGES}.pt"
    RUN_DIR="outputs/eval/text_location_eval/${CHECKPOINT_STEM}-${CHECKPOINT_HASH}/index-${EVAL_NUM_IMAGES}/gallery-${EVAL_GALLERY_SIZE}-seed-${EVAL_GALLERY_SEED}"
    mkdir -p "$RUN_DIR"

    echo "Evaluating checkpoint: $CHECKPOINT"
    echo "Index: $INDEX ($EVAL_NUM_IMAGES panoramas)"
    echo "Gallery size: $EVAL_GALLERY_SIZE, seed: $EVAL_GALLERY_SEED"

    python -m text_location_eval index \
        --checkpoint "$CHECKPOINT" --grid "$GRID" \
        --num-images "$EVAL_NUM_IMAGES" \
        --batch-size 8 --workers 4 --save-every 20000 \
        --output "$INDEX"

    python -m text_location_eval retrieve \
        --checkpoint "$CHECKPOINT" --grid "$GRID" \
        --num-images "$EVAL_NUM_IMAGES" \
        --index "$INDEX" \
        --gallery-size "$EVAL_GALLERY_SIZE" \
        --gallery-seed "$EVAL_GALLERY_SEED" \
        --output "$RUN_DIR/retrieval.json"

    # Separate processes release SigLIP GPU memory before vLLM loads.
    # The retrieval hash prevents conflicts with unrelated saved judgments.
    RETRIEVAL_HASH="$(sha256sum -- "$RUN_DIR/retrieval.json" | cut -c1-12)"
    JUDGE_DIR="$RUN_DIR/judge-$RETRIEVAL_HASH"

    python -m text_location_eval judge \
        --retrieval "$RUN_DIR/retrieval.json" --output "$JUDGE_DIR"

    python -m text_location_eval metrics \
        --retrieval "$RUN_DIR/retrieval.json" \
        --analysis "$JUDGE_DIR/query_analysis.jsonl" \
        --judgments "$JUDGE_DIR/judgments.jsonl" \
        --output "$JUDGE_DIR"

    echo "Evaluation results: $JUDGE_DIR"
elif [[ "$EVAL_ENABLED" != "0" ]]; then
    echo "EVAL_ENABLED must be 0 or 1" >&2
    exit 1
fi

date
    