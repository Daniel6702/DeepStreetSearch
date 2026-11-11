# LaVPR comparison (text-to-Street-View)

This adapter does not modify LaVPR or the existing DeepStreetSearch evaluation pipeline.
It uses the **same panorama IDs, source archives, queries, random controls, Qwen rubric,
and metric calculations** as `text_location_eval`. The LaVPR embeddings are computed
separately because DeepStreetSearch's embeddings are incompatible.

## Prerequisites

- `external/LaVPR` checked out, with `checkpoints/blip_lora_all_r64/adapter_model.safetensors`.
- DeepStreetSearch's normal Python `.venv` plus `peft` (`python -m pip install peft`).
- A completed DeepStreetSearch panorama index (`outputs/indexes/*.pt`) and its
  original image shards accessible at their saved `record['shard']` paths.
- The BLIP base model `Salesforce/blip-itm-base-coco` must be downloadable/cached.

Neither the MSLS nor GSV-Cities datasets are needed.

## Run (from the DeepStreetSearch root)

```bash
source .venv/bin/activate
sbatch external_eval/lavpr/run.bash outputs/indexes/base_index.pt 20000 42 max
```

Replace `base_index.pt` with **the same index used in the DeepStreetSearch
comparison**. `20000` and `42` must match its `--gallery-size` and
`--gallery-seed`. The optional last argument chooses `max` or `mean` view
aggregation; `max` is the default. The first step can take a while; it saves
resumable 256-panorama parts, and reuses the finished index on subsequent runs.

The run calls LaVPR `index` and `retrieve`, then the unchanged
`python -m text_location_eval judge` and `metrics` commands. Results are under
`outputs/eval/lavpr/`.

The adapter reproduces LaVPR's **cross-modal BLIP + LoRA** embedding path and
normalization, encoding each 360° panorama as the six perspective crops used by
DeepStreetSearch. `max` aggregation ranks each panorama by its highest-scoring
crop. `mean` aggregates all six crop embeddings before comparing with the query.
This panorama extension is **our evaluation adaptation**, not LaVPR's
original single-image MSLS evaluation.
