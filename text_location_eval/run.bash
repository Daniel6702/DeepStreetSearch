#!/bin/bash
#SBATCH --job-name=TextLocationEval
#SBATCH --partition=GPU24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%A.out
#SBATCH --error=R-%x.%A.err

set -euo pipefail

echo "Begin Setup"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate

CHECKPOINT="${1:-outputs/checkpoints/deepstreetsearch_E3.pt}"
NUM_IMAGES="${2:-1000000}"

CHECKPOINT_NAME="${CHECKPOINT##*/}"
CHECKPOINT_STEM="${CHECKPOINT_NAME%.*}"
CHECKPOINT_HASH="$(sha256sum -- "$CHECKPOINT" | cut -c1-12)"
RUN_DIR="outputs/eval/text_location_eval/deepstreetsearchFULL-17bfc1d79410-1000000"
 
#"${3:-outputs/eval/text_location_eval/${CHECKPOINT_STEM}-${CHECKPOINT_HASH}-${NUM_IMAGES}}"


echo "Begin Eval"

#Create index of image embeddings
python -m text_location_eval index \
  --checkpoint "$CHECKPOINT" --num-images "$NUM_IMAGES" \
  --output "$RUN_DIR/reference.pt" \
  --save-every 20000

#generate text embedding for each test prompt, get 10 best matching image embeddings
#python -m text_location_eval retrieve \
#  --checkpoint "$CHECKPOINT" --num-images "$NUM_IMAGES" \
#  --index "$RUN_DIR/reference.pt" --output "$RUN_DIR/retrieval.json"

#evalutate the retrieved locations
#python -m text_location_eval judge \
#  --retrieval "$RUN_DIR/retrieval.json" --output "$RUN_DIR"

#calculate evaluation results
#python -m text_location_eval metrics \
#  --retrieval "$RUN_DIR/retrieval.json" \
#  --analysis "$RUN_DIR/query_analysis.jsonl" \
#  --judgments "$RUN_DIR/judgments.jsonl" --output "$RUN_DIR"

echo "Eval Finished"
