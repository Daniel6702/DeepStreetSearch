"""Compare each image's embedding with the embedding of the Qwen description of that image."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import SIGLIP_MODEL
from evaluation.inference import load_model, encode_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--descriptions", default="results/descriptions.jsonl")
    p.add_argument("--checkpoint", default=None,
                   help="Trained checkpoint. Leave out for the untrained baseline")
    p.add_argument("--label", default=None,
                   help="Name for this run (default: 'trained' if --checkpoint given, else 'untrained')")
    p.add_argument("--grid", default="grids/world.aggrid")
    p.add_argument("--model", default=SIGLIP_MODEL)
    p.add_argument("--output", default="results/alignment")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Only use N pairs (0 = all)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class ImageDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        with Image.open(self.records[i]["image_path"]) as img:
            return img.convert("RGB")


@torch.inference_mode()
def embed_images(model, processor, records, device, batch_size, workers):
    loader = DataLoader(ImageDataset(records), batch_size=batch_size,
                        num_workers=workers, collate_fn=list)
    out = []
    for images in tqdm(loader, desc="Images"):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _, emb = model.encode_images(inputs, batch_size=len(images), num_views=1)
        if emb.shape[0] != len(images):
            raise RuntimeError(f"Expected {len(images)} embeddings, got {tuple(emb.shape)}")
        out.append(F.normalize(emb.float(), dim=-1).cpu())
        for im in images:
            im.close()
    return torch.cat(out)


@torch.inference_mode()
def embed_text(model, processor, texts, device, batch_size):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Text"):
        emb = encode_text(model, processor, texts[i:i + batch_size], device)
        out.append(F.normalize(emb.float(), dim=-1).cpu())
    return torch.cat(out)


def summarize(img, txt):
    sim = (img * txt).sum(-1)  # cosine similarity per image/description pair
    return {
        "n_pairs": len(sim),
        "mean_cosine_sim": sim.mean().item(),
        "mean_cosine_distance": (1 - sim).mean().item(),
        "std_sim": sim.std().item(),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or ("trained" if args.checkpoint else "untrained")

    records = []
    with open(args.descriptions, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if Path(r["image_path"]).is_file():
                records.append(r)
    if args.limit:
        records = records[:args.limit]
    texts = [r["description"] for r in records]
    print(f"{len(records)} image/description pairs, running: {label}")

    model, processor = load_model(args.checkpoint, args.grid, args.model, device)

    img = embed_images(model, processor, records, device, args.batch_size, args.workers)
    txt = embed_text(model, processor, texts, device, args.batch_size)

    torch.save({"ids": [r["id"] for r in records], "image": img, "text": txt},
               out_dir / f"{label}_embeddings.pt")

    result = summarize(img, txt)
    with open(out_dir / f"{label}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{label}:")
    for k, v in result.items():
        print(f"  {k:<22}{v:.4f}" if isinstance(v, float) else f"  {k:<22}{v}")

    base, trained = out_dir / "untrained.json", out_dir / "trained.json"
    if base.exists() and trained.exists():
        a, b = json.load(open(base)), json.load(open(trained))
        print(f"\n{'metric':<22}{'untrained':>12}{'trained':>12}")
        for k in a:
            if isinstance(a[k], float):
                print(f"{k:<22}{a[k]:>12.4f}{b[k]:>12.4f}")
    else:
        print("\n(Run again with --checkpoint to get the trained comparison.)")


if __name__ == "__main__":
    main()