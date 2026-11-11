import csv
import json

import torch

label = "untrained"  # change to "trained" for the other run

data = torch.load(f"results/alignment/{label}_embeddings.pt", weights_only=True)
ids, img, txt = data["ids"], data["image"], data["text"]

desc = {}
with open("results/descriptions.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        desc[r["id"]] = r["description"]

sim = (img * txt).sum(-1)

ranks = []
for s in range(0, len(txt), 1024):
    sims = txt[s:s + 1024] @ img.T
    rows = torch.arange(len(sims))
    true = sims[rows, rows + s]
    ranks.append((sims > true[:, None]).sum(1) + 1)
ranks = torch.cat(ranks)

out = f"results/alignment/{label}_per_image.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "cosine_sim", "txt2img_rank", "description"])
    for i, image_id in enumerate(ids):
        w.writerow([image_id, f"{sim[i]:.4f}", int(ranks[i]), desc[image_id]])

print(f"Saved {out}")