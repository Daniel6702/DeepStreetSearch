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
from modules.optimizers import separate_learning_rate_groups
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
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--geo-lr", type=float, default=None)
    parser.add_argument("--lr-scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.02, help="Fraction of total optimizer steps used for linear warmup (0 disables)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Global gradient norm clipping; 0 disables")
    parser.add_argument("--dataset-fraction", type=float, default=1.0, help="Fixed random panorama subset, (0, 1]")
    parser.add_argument("--seed", type=int, default=42, help="Seed for subset selection and per-epoch shuffling")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoints/deepstreetsearch.pt")
    parser.add_argument("--checkpoint-per-epoch", action="store_true",
                        help="Keep one checkpoint per epoch: NAME.epoch-01.pt, NAME.epoch-02.pt, ...")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-text-encoder", action="store_true",
                        help="Freeze only SigLIP2's text tower; train image tower and geographic head")
    parser.add_argument("--hierarchy-weights", type=float, nargs="+", help="Geographic loss weights, fine -> coarse")
    parser.add_argument("--geo-loss-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-loss-weight", type=float, default=2.0)
    parser.add_argument("--panorama-contrastive-weight", type=float, default=0.0,
                        help="Additional panorama-level text contrastive loss (0 disables)")
    parser.add_argument("--panorama-contrastive-aggregation", choices=["mean", "max"],
                        default="mean", help="Aggregate panorama views in the additional contrastive loss")
    parser.add_argument("--retrieval-fusion", choices=["none", "cross_view", "semantic_tokens"],
                        default="none", help="Experimental panorama representation for text retrieval")
    parser.add_argument("--fusion-layers", type=int, default=2,
                        help="Cross-view Transformer layers (default 2)")
    parser.add_argument("--fusion-heads", type=int, default=8,
                        help="Attention heads in experimental fusion (default 8)")
    parser.add_argument("--context-gate-init", type=float, default=0.05,
                        help="Initial residual strength for cross-view fusion (0,1)")
    parser.add_argument("--semantic-token-count", type=int, default=3,
                        help="Number of learned panorama tokens (default 3)")
    parser.add_argument("--semantic-token-loss-weight", type=float, default=0.0,
                        help="Text-to-learned-token contrastive weight (requires semantic_tokens fusion)")
    parser.add_argument("--geo-backbone-gradient-scale", type=float, default=0.0,
                        help="Geographic gradient entering the vision tower: 0=detached, 1=full")
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
    if args.backbone_lr is not None and args.backbone_lr <= 0:
        parser.error("--backbone-lr must be positive")
    if args.geo_lr is not None and args.geo_lr <= 0:
        parser.error("--geo-lr must be positive")
    if not 0 <= args.geo_view_dropout < 1:
        parser.error("--geo-view-dropout must be in [0, 1)")
    if not 0 <= args.geo_backbone_gradient_scale <= 1:
        parser.error("--geo-backbone-gradient-scale must be in [0, 1]")
    if args.panorama_contrastive_weight < 0:
        parser.error("--panorama-contrastive-weight cannot be negative")
    if min(args.fusion_layers, args.fusion_heads, args.semantic_token_count) < 1:
        parser.error("--fusion-layers, --fusion-heads and --semantic-token-count must be positive")
    if not 0 < args.context_gate_init < 1:
        parser.error("--context-gate-init must be in (0,1)")
    if args.semantic_token_loss_weight < 0:
        parser.error("--semantic-token-loss-weight cannot be negative")
    if (args.retrieval_fusion == "semantic_tokens") != (args.semantic_token_loss_weight > 0):
        parser.error("Semantic tokens require --semantic-token-loss-weight > 0, "
                     "and that loss requires --retrieval-fusion semantic_tokens")
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
        geo_backbone_gradient_scale=args.geo_backbone_gradient_scale,
        retrieval_fusion=args.retrieval_fusion,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        semantic_token_count=args.semantic_token_count,
        context_gate_init=args.context_gate_init,
    )

    loss_fn = DeepStreetSearchLoss(
        grid,
        hierarchy_weights=args.hierarchy_weights,
        geographic_weight=args.geo_loss_weight,
        contrastive_weight=args.contrastive_loss_weight,
        revised_contrastive_positives=args.revised_contrastive_positives,
        panorama_contrastive_weight=args.panorama_contrastive_weight,
        panorama_contrastive_aggregation=args.panorama_contrastive_aggregation,
        semantic_token_loss_weight=args.semantic_token_loss_weight,
    )

    trainer = Trainer(model, loader, loss_fn, device=args.device)
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.learning_rate
    geo_lr = args.geo_lr if args.geo_lr is not None else args.learning_rate
    optimizer_parameters = (
        separate_learning_rate_groups(trainer.model, backbone_lr, geo_lr)
        if args.backbone_lr is not None or args.geo_lr is not None
        else (parameter for parameter in trainer.model.parameters() if parameter.requires_grad)
    )
    optimizer = AdamW(
        optimizer_parameters,
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
        "panorama_contrastive_weight": args.panorama_contrastive_weight,
        "panorama_contrastive_aggregation": args.panorama_contrastive_aggregation,
        "retrieval_fusion": args.retrieval_fusion,
        "fusion_layers": args.fusion_layers,
        "fusion_heads": args.fusion_heads,
        "semantic_token_count": args.semantic_token_count,
        "context_gate_init": args.context_gate_init,
        "semantic_token_loss_weight": args.semantic_token_loss_weight,
        "geo_backbone_gradient_scale": args.geo_backbone_gradient_scale,
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
                "panorama_contrastive_weight": 0.0,
                "panorama_contrastive_aggregation": "mean",
                "geo_backbone_gradient_scale": 0.0,
                "retrieval_fusion": "none",
                "fusion_layers": 2,
                "fusion_heads": 8,
                "semantic_token_count": 3,
                "context_gate_init": 0.05,
                "semantic_token_loss_weight": 0.0,
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
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        dataset.set_epoch(start_epoch)
        print(f"Resuming from epoch {start_epoch + 1} (optimizer step {trainer.global_step})")

    print(f"Grid classes fine -> coarse: {grid.n_classes}")
    print(f"Descriptions: {description_path}")
    print(f"Description mode: {args.description_mode}")
    print(f"Revised contrastive positives: {args.revised_contrastive_positives}")
    print(f"Frozen text encoder: {args.freeze_text_encoder}")
    print(f"Geographic branch: {args.geo_head} head, {args.geo_pooling} pooling, "
          f"view dropout {args.geo_view_dropout:.0%}; "
          f"geo->vision gradient scale {args.geo_backbone_gradient_scale:g}")
    print(f"Panorama contrastive loss: weight {args.panorama_contrastive_weight:g}, "
          f"aggregation {args.panorama_contrastive_aggregation}")
    print(f"Retrieval fusion: {args.retrieval_fusion}, layers={args.fusion_layers}, "
          f"heads={args.fusion_heads}, gate-init={args.context_gate_init:g}, "
          f"tokens={args.semantic_token_count}, token loss weight={args.semantic_token_loss_weight:g}")
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
            f"contrastive: {metrics['contrastive']:.4f} - "
            f"panorama: {metrics['panorama_contrastive']:.4f} - "
            f"semantic: {metrics['semantic_contrastive']:.4f}"
        )
        checkpoint_path = Path(args.checkpoint)
        if args.checkpoint_per_epoch:
            checkpoint_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}.epoch-{epoch + 1:02d}{checkpoint_path.suffix}"
            )
            if checkpoint_path.exists():
                raise FileExistsError(f"Refusing to overwrite preserved epoch: {checkpoint_path}")
        trainer.save_checkpoint(optimizer, epoch, checkpoint_path, scheduler=scheduler, run_config=run_config)
        if args.checkpoint_per_epoch:
            print(f"Saved: epoch checkpoint {checkpoint_path}")

if __name__ == "__main__":
    main()
