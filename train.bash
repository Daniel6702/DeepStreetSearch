#!/bin/bash
#SBATCH --job-name=DeepStreetSearch
#SBATCH --partition=GPU24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
# SBATCH --nodelist=node4
#SBATCH --output=R-%x.%A.out
#SBATCH --error=R-%x.%A.err

set -e

cd ~/DeepStreetSearch
source .venv/bin/activate

date
nvidia-smi

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python -u train.py \
    --grid outputs/grids/world.aggrid \
    --dataset-fraction 1.0 \
    --seed 42 \
    --epochs 10 \
    --batch-size 4 \
    --workers 8 \
    --description-mode random \
    --learning-rate 1e-5 \
    --lr-scheduler cosine \
    --warmup-ratio 0.02 \
    --max-grad-norm 1.0 \
    --checkpoint checkpoints/deepstreetsearchFULL.pt

date
    