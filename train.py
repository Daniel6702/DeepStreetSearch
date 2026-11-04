from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.optim import AdamW

from adaptive_geogrid import load_grid
from config import SIGLIP_MODEL, DATASET_PATH
from modules.dataset import PanoramaDataset
from modules.losses import DeepStreetSearchLoss
from modules.model import GeoSearchModel
from modules.trainer import Trainer, make_scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=f"{DATASET_PATH}/world")
    parser.add_argument("--grid", default="outputs/grids/world.aggrid")
    parser.add_argument("--geo-label-cache", default="outputs/grids/world_geo_labels.npz", help="Load geographic labels from this file, or create it if missing")
    parser.add_argument("--descriptions", default=None, help="Description SQLite file or per-shard database directory")
    parser.add_argument("--model", default=SIGLIP_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.02, help="Fraction of total optimizer steps used for linear warmup (0 disables)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Global gradient norm clipping; 0 disables")
    parser.add_argument("--dataset-fraction", type=float, default=1.0, help="Fixed random panorama subset, (0, 1]")
    parser.add_argument("--seed", type=int, default=42, help="Seed for subset selection and per-epoch shuffling")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoints/deepstreetsearch.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-text-encoder", action="store_true",
                        help="Freeze only SigLIP2's text tower; train image tower and geographic head")
    parser.add_argument("--hierarchy-weights", type=float, nargs="+", help="Geographic loss weights, fine -> coarse")
    parser.add_argument("--geo-loss-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-loss-weight", type=float, default=2.0)
    parser.add_argument("--geo-head", choices=["linear", "mlp"], default="mlp",
                        help="Geographic classifier architecture (default: mlp)")
    parser.add_argument("--geo-pooling", choices=["mean", "attention"], default="attention",
                        help="How panorama crops are combined for geographic prediction")
    parser.add_argument("--geo-view-dropout", type=float, default=0.2,
                        help="Probability of excluding each crop from geographic pooling during training")
    parser.add_argument(
        "--revised-contrastive-positives", action="store_true",
        help="Ignore other crops from the positive panorama as contrastive negatives (default: off)",
    )
    parser.add_argument("--prompts", type=int, nargs="+", choices=[1, 2, 3, 4], help="Description prompts to train on; default is all")
    parser.add_argument(
        "--description-mode",
        choices=["all", "random"],
        default="all",
        help="all: train on every selected description per crop (default); random: sample one per crop per batch",
    )
    parser.add_argument("--attn-implementation", default="sdpa", choices=["eager", "sdpa"])
    args = parser.parse_args()
    if not 0 < args.dataset_fraction <= 1:
        parser.error("--dataset-fraction must be in (0, 1]")
    if not 0 <= args.warmup_ratio <= 1:
        parser.error("--warmup-ratio must be in [0, 1]")
    if args.max_grad_norm < 0:
        parser.error("--max-grad-norm cannot be negative")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers cannot be negative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 <= args.geo_view_dropout < 1:
        parser.error("--geo-view-dropout must be in [0, 1)")
    return args


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    grid = load_grid(args.grid)
    description_path = Path(args.descriptions) if args.descriptions else Path(args.dataset) / "descriptions"
    prompt_indices = None if args.prompts is None else [prompt - 1 for prompt in args.prompts]

    dataset = PanoramaDataset(
        args.dataset,
        shuffle=True,
        grid=grid,
        hierarchy_cache=args.geo_label_cache,
        fraction=args.dataset_fraction,
        seed=args.seed,
    )

    loader = dataset.loader(
        model_name=args.model,
        descriptions=description_path,
        prompt_indices=prompt_indices,
        description_mode=args.description_mode,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.device.startswith("cuda") and torch.cuda.is_available(),
    )

    model = GeoSearchModel(
        grid,
        model_name=args.model,
        freeze_backbone=args.freeze_backbone,
        freeze_text_encoder=args.freeze_text_encoder,
        attn_implementation=args.attn_implementation,
        geo_head=args.geo_head,
        geo_pooling=args.geo_pooling,
        geo_view_dropout=args.geo_view_dropout,
    )

    loss_fn = DeepStreetSearchLoss(
        grid,
        hierarchy_weights=args.hierarchy_weights,
        geographic_weight=args.geo_loss_weight,
        contrastive_weight=args.contrastive_loss_weight,
        revised_contrastive_positives=args.revised_contrastive_positives,
    )

    trainer = Trainer(model, loader, loss_fn, device=args.device)
    optimizer = AdamW(
        (parameter for parameter in trainer.model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    steps_by_epoch = [
        dataset.steps_per_epoch(args.batch_size, args.workers, epoch)
        for epoch in range(args.epochs)
    ]
    total_steps = sum(steps_by_epoch)
    if total_steps == 0:
        raise ValueError("No training batches available for the selected dataset fraction")
    scheduler, warmup_steps = make_scheduler(
        optimizer, total_steps, args.warmup_ratio, args.lr_scheduler
    )
    run_config = {
        "dataset_fraction": args.dataset_fraction,
        "num_samples": dataset.num_samples,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": min(args.workers, len(dataset.shards)),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "freeze_backbone": args.freeze_backbone,
        "freeze_text_encoder": args.freeze_text_encoder,
        "geo_loss_weight": args.geo_loss_weight,
        "contrastive_loss_weight": args.contrastive_loss_weight,
        "lr_scheduler": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "description_mode": args.description_mode,
        "revised_contrastive_positives": args.revised_contrastive_positives,
        "geo_head": args.geo_head,
        "geo_pooling": args.geo_pooling,
        "geo_view_dropout": args.geo_view_dropout,
        "max_grad_norm": args.max_grad_norm,
        "prompts": args.prompts,
    }

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        if checkpoint.get("run_config") is not None:
            legacy_defaults = {
                "revised_contrastive_positives": False,
                "geo_head": "linear",
                "geo_pooling": "mean",
                "geo_view_dropout": 0.0,
                "freeze_text_encoder": False,
                # Older checkpoints didn't record these settings. They are
                # intentionally not enforced for legacy resumes.
            }
            optional_legacy_keys = {"freeze_backbone", "geo_loss_weight", "contrastive_loss_weight"}
            changed = [
                key for key, value in run_config.items()
                if key not in optional_legacy_keys or key in checkpoint["run_config"]
                if checkpoint["run_config"].get(key, legacy_defaults.get(key)) != value
            ]
            if changed:
                raise ValueError(
                    f"Training settings differ from checkpoint ({', '.join(changed)}). "
                    "Resume with the original settings to preserve the schedule and subset."
                )
        elif args.lr_scheduler != "none" or args.warmup_ratio != 0:
            raise ValueError(
                "Legacy checkpoint has no scheduler settings. "
                "Resume with --lr-scheduler none --warmup-ratio 0, or start a new run."
            )
        trainer.model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch > args.epochs:
            raise ValueError("Checkpoint epoch exceeds --epochs")
        expected_step = sum(steps_by_epoch[:start_epoch])
        trainer.global_step = checkpoint.get("global_step", expected_step)
        if trainer.global_step != expected_step:
            raise ValueError(
                f"Checkpoint step {trainer.global_step} differs from expected {expected_step} "
                "for completed epochs"
            )
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
            if scheduler.last_epoch != trainer.global_step:
                raise ValueError("Checkpoint learning-rate scheduler position does not match optimizer steps")
        elif args.lr_scheduler != "none" or args.warmup_ratio != 0:
            raise ValueError("Checkpoint is missing the learning-rate scheduler state")
        else:
            # Legacy constant-LR checkpoint: keep subsequent checkpoints aligned.
            scheduler.last_epoch = trainer.global_step
            scheduler._step_count = trainer.global_step + 1
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        dataset.set_epoch(start_epoch)
        print(f"Resuming from epoch {start_epoch + 1} (optimizer step {trainer.global_step})")

    print(f"Grid classes fine -> coarse: {grid.n_classes}")
    print(f"Descriptions: {description_path}")
    print(f"Description mode: {args.description_mode}")
    print(f"Revised contrastive positives: {args.revised_contrastive_positives}")
    print(f"Frozen text encoder: {args.freeze_text_encoder}")
    print(f"Geographic branch: {args.geo_head} head, {args.geo_pooling} pooling, "
          f"view dropout {args.geo_view_dropout:.0%}; geographic gradients do not reach the backbone")
    print(f"Dataset: {dataset.num_samples:,}/{len(dataset.metadata):,} panoramas ({args.dataset_fraction:.2%}), seed {args.seed}")
    print(f"LR schedule: {args.lr_scheduler}, warmup {warmup_steps:,}/{total_steps:,} optimizer steps")
    print(f"Max gradient norm: {args.max_grad_norm or 'disabled'}")

    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        metrics = trainer.train_epoch(
            optimizer,
            scheduler=scheduler,
            max_grad_norm=args.max_grad_norm,
            expected_steps=steps_by_epoch[epoch],
            expected_samples=dataset.num_samples,
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"loss: {metrics['loss']:.4f} - "
            f"geo: {metrics['geographic']:.4f} - "
            f"contrastive: {metrics['contrastive']:.4f}"
        )
        trainer.save_checkpoint(optimizer, epoch, args.checkpoint, scheduler=scheduler, run_config=run_config)

if __name__ == "__main__":
    main()
