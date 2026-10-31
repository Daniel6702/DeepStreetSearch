# Text-to-location evaluation (independent development framework)

This package does **not** import or modify `eval.py` or `evaluation/`. It evaluates
free-form text queries against a fixed, georeferenced Street View panorama gallery.
The first version supports only the trained DeepStreetSearch checkpoint (no other
model comparison). Its 100 engineered queries have no single correct GPS location.

## Run on the GPU24 cluster

From the project root, after training has produced the checkpoint:

```bash
sbatch text_location_eval/run.bash
```

The job runs **four separate processes**, releasing SigLIP before starting Qwen3-VL:

```bash
python -m text_location_eval index --checkpoint checkpoints/deepstreetsearchFULL.pt --num-images 5000
python -m text_location_eval retrieve --checkpoint checkpoints/deepstreetsearchFULL.pt --num-images 5000
python -m text_location_eval judge
python -m text_location_eval metrics
```

Optional script arguments: `sbatch text_location_eval/run.bash CHECKPOINT NUM_IMAGES RUN_DIR`.
By default, the run directory includes the checkpoint filename and a short
SHA-256 hash of its **contents** (for example,
`outputs/eval/text_location_eval/deepstreetsearchFULL-a1b2c3d4e5f6/`). Repeating
a run with unchanged weights reuses the index; replacing the weights produces
a new directory automatically. Passing `RUN_DIR` overrides this default.
The dataset path comes from `config.DATASET_PATH` and defaults to `<DATASET_PATH>/world`.
Override it with `--dataset` on the `index` and `retrieve` commands if needed.
For a quick smoke test, use `retrieve --limit-queries 3` and **a separate output
directory** for the retrieval and judge results. Run `python -m text_location_eval
<command> --help` for additional controls.

## What each stage does

1. **index**: reproducibly samples panorama IDs (`--seed 42`) from the *training*
   dataset; splits each panorama into six crops using the existing project code;
   encodes all views with the selected checkpoint. Caches half-precision crop
   embeddings, geographic coordinates, source archive paths, and SHA-256
   fingerprints for the checkpoint, grid, and metadata. If anything differs,
   a cached index is rejected instead of silently reused (`--rebuild` to replace).
   For large galleries, progress is saved atomically every 5,000 panoramas
   (configurable with `--save-every`) into `reference.parts/part-*.pt`.
   If interrupted, rerun the **same index command**: previously saved panoramas
   are excluded before decoding, so only the unfinished images are encoded.
   After all parts exist, they are merged into the usual `reference.pt` format.
   Parts are retained for recovery and temporarily use additional disk space.
   `--rebuild` intentionally discards saved parts and starts again.
2. **retrieve**: embeds all 100 text-only queries, returns the top ten locations
   using cosine similarity to the L2-normalized mean of the six crop features
   (or `--aggregation max` for best-matching view), and samples five random
   control panoramas per query. It flags queries truncated by the text tokenizer.
3. **judge**: builds six-view contact sheets only for retrieved/control images.
   Qwen3-VL first identifies relevant categories using **only the query**;
   the resulting category mask/importance remains fixed for every image.
   Blinded visual judging assigns 0-3 to each relevant category and a separate
   holistic 0-3 relevance grade. It never receives the retrieval rank, model
   similarity, coordinates, or source location metadata. JSON results are
   validated and incrementally saved; rerunning resumes incomplete judgments.
   Equivalent numeric score formats are normalized. Unresolved model responses
   are logged and skipped without stopping the remaining evaluations.
4. **metrics**: computes judged-pool nDCG@10, P@5, P@10 (grade >= 2),
   Success@10, mean holistic relevance, mean weighted category relevance,
   random-control relevance, and per-category averages.

Results are written to the checkpoint-specific run directory by default
(the individual Python commands above still default to `outputs/eval/text_location_eval/dev/`):

- `reference.pt` — checkpoint-specific image index
- `retrieval.json` — prompts, candidate locations, similarities, control IDs
- `query_analysis.jsonl` — category relevance and weights for each query
- `judgments.jsonl` — one row per evaluated text/panorama pair
- `metrics.json`, `per_query.csv`, `per_category.csv` — quantitative results
- `previews/` — six-view contact sheets used by the judge

## Interpretation and limitations

- **The initial 5,000-reference gallery is sparse.** A poor result could mean
  the requested scene does not exist in the sampled gallery. This is a pipeline
  development experiment, not a final unseen-location result.
- The 100 prompts are synthetic; they were written independently of selected
  training panoramas, but they do not have annotated ground-truth matches.
- Visual category scores describe scene correspondence, **not the probability
  of a correct geographic location**. Place names are explicitly excluded from
  VLM visual judgment. Geocoding and independent geographic consistency checks
  are future work.
- `nDCG@10` is computed with grades for the retrieved top ten and random
  controls, so the ideal ranking is only over this **limited judged pool**.
  High nDCG can coexist with poor absolute relevance. Meaningful comparisons
  with other models will require a *shared, independently judged* candidate
  pool; P@K and mean relevance should always be reported alongside nDCG.
- The judge is a VLM, not human ground truth. A human-rated validation sample
  is needed before using these figures as final research evidence.
- When an output remains invalid after retries, its details are saved in
  judge_failures.jsonl and it can be retried on the next run. Metric averages
  use only complete queries; the summary reports judgment coverage and a
  partial status rather than inventing zero scores for missing judgments.

Focused tests (no models, data, or GPU required):

```bash
python -m unittest discover -s text_location_eval/tests -v
```
