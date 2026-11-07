"""Describe a folder of images with Qwen3-VL and store descriptions linked by image ID."""

import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from config import (
    MODEL, MAX_TOKENS, MAX_MODEL_LENGTH,
    TEMPERATURE, TOP_P, REPETITION_PENALTY, PROMPT1,
)
from modules.panorama import split_panorama
from modules.lvlm import QwenVLBackend

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="Folder with the Kaggle images")
    p.add_argument("--output", default="results/descriptions.jsonl")
    p.add_argument("--batch-images", type=int, default=4, help="Images per vLLM call")
    p.add_argument("--limit", type=int, default=0, help="Only process N images (0 = all)")
    p.add_argument("--no-split", action="store_true",
                   help="Describe the whole image as one view (use if images are not 360 panoramas)")
    return p.parse_args()


def find_images(root):
    root = Path(root)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS)
    # ID = relative path without extension, so it stays unique across subfolders
    return [(p.relative_to(root).with_suffix("").as_posix(), p) for p in paths]


def load_done_ids(output):
    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass  # ignore a half-written last line
    return done


def get_views(path, split):
    with Image.open(path) as img:
        image = img.convert("RGB")
    if not split:
        return [image]
    try:
        return [crop.image for crop in split_panorama(image)]
    finally:
        image.close()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    items = find_images(args.images)
    done = load_done_ids(output)
    todo = [(i, p) for i, p in items if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(items)} images found, {len(done)} already done, {len(todo)} to process")

    backend = QwenVLBackend(model_name=MODEL, max_model_len=MAX_MODEL_LENGTH)

    with output.open("a", encoding="utf-8") as out:
        for start in tqdm(range(0, len(todo), args.batch_images), desc="Describing"):
            batch = todo[start:start + args.batch_images]
            all_views, owners = [], []  # owners[i] = index into batch

            for b, (image_id, path) in enumerate(batch):
                try:
                    views = get_views(path, split=not args.no_split)
                except Exception as e:
                    print(f"Skipping {image_id}: {e}")
                    continue
                all_views.extend(views)
                owners.extend([b] * len(views))

            if not all_views:
                continue

            try:
                responses = backend.generate_batch(
                    images=all_views,
                    prompt=PROMPT1,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=REPETITION_PENALTY,
                )
            finally:
                for v in all_views:
                    v.close()

            per_image = {}
            for owner, response in zip(owners, responses):
                per_image.setdefault(owner, []).append(response.text)

            for b, views in per_image.items():
                image_id, path = batch[b]
                record = {"id": image_id, "image_path": str(path)}
                if args.no_split:
                    record["description"] = views[0]
                else:
                    record["views"] = views
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    print(f"Done. Saved to {output}")


if __name__ == "__main__":
    main()