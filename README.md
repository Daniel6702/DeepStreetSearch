# DeepStreetSearch

DeepStreetSearch trains a shared SigLIP image/text representation for geographic retrieval while also teaching the visual encoder geographic discrimination with an adaptive hierarchical grid.

## Training objective

Each panorama is split into the configured views. Their SigLIP visual features are averaged into one panorama representation.

Training uses two losses:

- **Contrastive loss:** SigLIP's native pairwise sigmoid loss between the panorama embedding and one randomly sampled stored description for that panorama.
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
    --boundary grids/ne_10m_land.geojson \
    --output grids/world.aggrid
```

## Train

```bash
python train.py \
    --dataset datasets/world \
    --grid grids/world.aggrid \
    --batch-size 4 \
    --workers 4
```

Useful options:

```text
--prompts 1 2 3 4
--hierarchy-weights 1 1 1 1 1 1
--geo-loss-weight 1.0
--contrastive-loss-weight 1.0
--freeze-backbone
--resume checkpoints/deepstreetsearch.pt
```

The default retrieval backbone is `google/siglip2-base-patch16-512`.
