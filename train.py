from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from adaptive_geogrid import load_grid
from config import SIGLIP_MODEL, DATASET_PATH
from modules.dataset import PanoramaDataset
from modules.losses import DeepStreetSearchLoss
from modules.model import GeoSearchModel
from modules.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=f"{DATASET_PATH}/world")
    parser.add_argument("--grid", default="grids/world.aggrid")
    parser.add_argument("--geo-label-cache", default="grids/world_geo_labels.npz", help="Load geographic labels from this file, or create it if missing")
    parser.add_argument("--descriptions", default=f"{DATASET_PATH}/world/descriptions", help="Description SQLite file or per-shard database directory")
    parser.add_argument("--model", default=SIGLIP_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoints/deepstreetsearch.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--hierarchy-weights", type=float, nargs="+", help="Geographic loss weights, fine -> coarse")
    parser.add_argument("--geo-loss-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-loss-weight", type=float, default=1.0)
    parser.add_argument("--prompts", type=int, nargs="+", choices=[1, 2, 3, 4], help="Description prompts to train on; default is all")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["eager", "sdpa"])
    return parser.parse_args()


def main():
    args = parse_args()

    grid = load_grid(args.grid)
    description_path = Path(args.descriptions) if args.descriptions else Path(args.dataset) / "descriptions"
    prompt_indices = None if args.prompts is None else [prompt - 1 for prompt in args.prompts]

    dataset = PanoramaDataset(
        args.dataset,
        shuffle=True,
        grid=grid,
        hierarchy_cache=args.geo_label_cache,
    )

    loader = dataset.loader(
        model_name=args.model,
        descriptions=description_path,
        prompt_indices=prompt_indices,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.device.startswith("cuda") and torch.cuda.is_available(),
    )

    model = GeoSearchModel(
        grid,
        model_name=args.model,
        freeze_backbone=args.freeze_backbone,
        attn_implementation=args.attn_implementation,
    )

    loss_fn = DeepStreetSearchLoss(
        grid,
        hierarchy_weights=args.hierarchy_weights,
        geographic_weight=args.geo_loss_weight,
        contrastive_weight=args.contrastive_loss_weight,
    )

    trainer = Trainer(model, loader, loss_fn, device=args.device)
    optimizer = AdamW(
        (parameter for parameter in trainer.model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    print(f"Grid classes fine -> coarse: {grid.n_classes}")
    print(f"Descriptions: {description_path}")

    for epoch in range(start_epoch, args.epochs):
        metrics = trainer.train_epoch(optimizer)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"loss: {metrics['loss']:.4f} - "
            f"geo: {metrics['geographic']:.4f} - "
            f"contrastive: {metrics['contrastive']:.4f}"
        )
        trainer.save_checkpoint(optimizer, epoch, args.checkpoint)


if __name__ == "__main__":
    main()