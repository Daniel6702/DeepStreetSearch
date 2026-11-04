#!/bin/bash
#SBATCH --job-name=ImageLocationEval
#SBATCH --partition=GPU24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%A.out
#SBATCH --error=R-%x.%A.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate

CHECKPOINT="${1:-outputs/checkpoints/deepstreetsearch_E3.pt}"
TEST_DATASET="${2:-/home/austen/Street-View-Harvester/datasets/world}"
SPLIT="${3:-heldout}" # heldout (fraction < 1), external (separate dataset), or all (diagnostic only)
NUM_IMAGES="${4:-2000}"
GRID="outputs/grids/world.aggrid"

NAME="$(basename "${CHECKPOINT%.pt}")"
HASH="$(sha256sum -- "$CHECKPOINT" | cut -c1-12)"
OUTPUT="outputs/eval/image_location_eval/${NAME}-${HASH}/${SPLIT}-${NUM_IMAGES}"

python -m image_location_eval \
    --checkpoint "$CHECKPOINT" \
    --grid "$GRID" \
    --dataset "$TEST_DATASET" \
    --split "$SPLIT" \
    --num-images "$NUM_IMAGES" \
    --batch-size 8 --workers 4 \
    --output "$OUTPUT"
