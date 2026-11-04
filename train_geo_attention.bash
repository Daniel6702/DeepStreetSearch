#!/bin/bash
#SBATCH --job-name=GeoAttention
#SBATCH --partition=GPU24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%A.out
#SBATCH --error=R-%x.%A.err

set -euo pipefail
cd ~/DeepStreetSearch
source .venv/bin/activate

CHECKPOINT="outputs/checkpoints/geo_attention_5P_E8.pt"

python -u train.py \
    --grid outputs/grids/world.aggrid \
    --geo-label-cache outputs/grids/world_geo_labels.npz \
    --dataset-fraction 0.05 \
    --seed 42 --epochs 8 --batch-size 4 --workers 8 \
    --description-mode random \
    --learning-rate 1e-5 --lr-scheduler cosine \
    --warmup-ratio 0.02 --max-grad-norm 1.0 \
    --geo-loss-weight 1.0 --contrastive-loss-weight 2.0 \
    --geo-head mlp --geo-pooling attention --geo-view-dropout 0.0 \
    --checkpoint "$CHECKPOINT"

sbatch text_location_eval/eval_checkpoint.bash "$CHECKPOINT"
sbatch image_location_eval/run.bash "$CHECKPOINT"