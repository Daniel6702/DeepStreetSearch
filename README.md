# DeepStreetSearch

DeepStreetSearch trains a shared SigLIP image/text representation for geographic retrieval while also teaching the visual encoder geographic discrimination with an adaptive hierarchical grid.

## Training objective

Each panorama is split into the configured views. Their SigLIP visual features are averaged into one panorama representation.

Training uses two losses:

- **Contrastive loss:** SigLIP's pairwise sigmoid loss between each image crop and its descriptions. By default, all four descriptions of each crop are positive pairs for that crop; other crops are negatives. Alternatively, randomly sample one of the available descriptions for each crop per batch.
- **Geographic loss:** the same fine-to-coarse hierarchical loss used by DeepGeoGuesser. Fine-cell probabilities are aggregated through the adaptive grid hierarchy and supervised at every level.

The default total loss is:

```text
loss = geographic_loss + contrastive_loss
```

Both terms can be reweighted from the command line.

## Dataset layout

```text
datasets/world/
├── metadata.csv
├── images/
│   ├── shard_00000.tar
│   └── ...
└── descriptions/
    ├── shard_00000.tar.sqlite
    └── ...
```

The description filenames match the per-shard databases produced by `tools/description_worker.py`.

## Build the grid

```bash
python tools/build_grid.py \
    --dataset datasets/world \
    --boundary outputs/grids/ne_10m_land.geojson \
    --output outputs/grids/world.aggrid
```

## Train

```bash
python train.py \
    --dataset datasets/world \
    --grid outputs/grids/world.aggrid \
    --batch-size 4 \
    --workers 4
```

Description training modes:

```bash
# Default: all four descriptions per crop are positive pairs.
python train.py --description-mode all

# Random: one of the four descriptions is selected independently per crop and batch.
python train.py --description-mode random
```

Both modes accept `--prompts` to limit the eligible descriptions (e.g., `--prompts 1 3` uses both as positives in `all` mode, or samples one of those two in `random` mode). In `random` mode, descriptions are resampled when each batch is collated, including across epochs. Image crops and geographic supervision are unchanged.


Training on a reproducible 5% subset with cosine LR decay, linear warmup, and gradient clipping:

```bash
python train.py \
    --dataset datasets/world \
    --grid outputs/grids/world.aggrid \
    --dataset-fraction 0.05 \
    --seed 42 \
    --epochs 10 \
    --batch-size 4 \
    --description-mode random \
    --lr-scheduler cosine \
    --warmup-ratio 0.02 \
    --max-grad-norm 1.0
```

`--dataset-fraction` selects the same panorama IDs once using `--seed` (minimum one panorama), then shuffles the selected images differently each epoch. The sampler skips decoding unselected images; it still scans archive headers. Fractions are measured by panorama, not crop. Batch/optimizer-step counts account for partial batches from separate DataLoader workers, so the LR schedule spans the actual total optimizer updates. The default `--dataset-fraction 1.0` uses every panorama.

The default LR policy is cosine decay with 2% linear warmup; clipping limits the global gradient norm to `1.0`. All are independent controls: `--lr-scheduler none` selects a constant post-warmup LR, `--warmup-ratio 0` disables warmup, and `--max-grad-norm 0` disables clipping. The scheduler is updated **after** each optimizer step, and checkpoints save its state and the run settings. To resume, supply `--resume` with the same dataset-fraction, seed, epochs, worker count, batch size, and LR-related settings. Old checkpoints without scheduler state can only be resumed with `--lr-scheduler none --warmup-ratio 0`.

Useful options:

```text
--description-mode all|random
--dataset-fraction 1.0
--seed 42
--lr-scheduler cosine|none
--warmup-ratio 0.02
--max-grad-norm 1.0
--prompts 1 2 3 4
--hierarchy-weights 1 1 1 1 1 1
--geo-loss-weight 1.0
--contrastive-loss-weight 1.0
--freeze-backbone
--resume checkpoints/deepstreetsearch.pt
```

The default retrieval backbone is `google/siglip2-base-patch16-512`.
