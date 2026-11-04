# Image-to-location evaluation

Measures geographic head predictions against known panorama coordinates without a
reference embedding index, text prompts, or LLM judgments. Evaluates all six
panorama crops through the checkpoint's configured mean/attention aggregation.

From the repository root:

```bash
python -m image_location_eval \
  --checkpoint outputs/checkpoints/model.pt \
  --grid outputs/grids/world.aggrid \
  --num-images 2000 \
  --output outputs/eval/image_location_eval/model
```

Or submit `sbatch image_location_eval/run.bash CHECKPOINT TEST_DATASET SPLIT NUM_IMAGES`.
The default `heldout` split **reconstructs the training sample** using the
checkpoint's `dataset_fraction` and `seed`, excludes those panorama IDs, and
selects a deterministic test subset. It works only for checkpoints trained on
less than 100% of that dataset. For models trained on 100%, provide a **different
test dataset** and use `--split external` (and `--train-dataset` if training
data is not the default world dataset). Duplicate panorama IDs are rejected.

`--split all` intentionally includes training images and is useful for
diagnostics only; do not report it as held-out performance.

Outputs:

- `summary.json`: median/mean/p75/p90 distance (km); accuracy within
  1/10/25/100/500/2500 km; hierarchical cell accuracy; fine-cell top-1/5/10;
  and best top-k geographic distance.
- `predictions.csv`: per-panorama true and predicted coordinates, cell IDs,
  distance error and top-k results.

Predicted coordinates are representative points of the predicted grid cells,
not continuously regressed coordinates. Distance error is therefore bounded
by grid resolution even for a correct fine-cell classification.
