"""Recompute broken cross-view vectors from an index's valid raw crop vectors.

No dataset decoding, image encoder, or Slurm GPU is needed. The input index is
never overwritten. Example:

python -m tools.repair_cross_view_index \
  --checkpoint outputs/checkpoints/fusion_cross_view_10P_E4.pt \
  --index outputs/indexes/fusion_cross_view_10P_E4-<hash>-test-20000.pt \
  --output outputs/indexes/fusion_cross_view_10P_E4-<hash>-test-20000-repaired.pt
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch

from modules.inference import _file_sha256
from modules.model import CrossViewTransformer


def repair(checkpoint_path: Path, index_path: Path, output: Path,
           batch_size: int, device: str) -> None:
    if index_path.resolve() == output.resolve():
        raise ValueError("The repaired index must have a different path from the original")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing index: {output}")
    if batch_size < 1:
        raise ValueError("batch-size must be positive")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config") or {}
    if config.get("retrieval_fusion") != "cross_view":
        raise ValueError("Checkpoint is not a cross-view fusion model")

    index = torch.load(index_path, map_location="cpu", weights_only=True)
    if index.get("signature", {}).get("checkpoint_sha256") != _file_sha256(checkpoint_path):
        raise ValueError("Index signature does not match the checkpoint")
    original = index.get("raw_embeddings")
    corrupted = index.get("embeddings")
    if (not torch.is_tensor(original) or not torch.is_tensor(corrupted)
            or original.ndim != 3 or original.shape[1] != 6
            or corrupted.shape != original.shape
            or len(index.get("records", [])) != original.shape[0]):
        raise ValueError("Expected matching [panoramas, 6, dim] raw and contextual embeddings")
    if not torch.isfinite(original).all():
        raise ValueError("Raw embeddings contain NaN/Inf and cannot repair this index")

    prefix = "cross_view_fusion."
    state = {key[len(prefix):]: tensor for key, tensor in checkpoint["model"].items()
             if key.startswith(prefix)}
    if not state:
        raise ValueError("Checkpoint contains no cross-view fusion weights")
    fusion = CrossViewTransformer(
        dim=original.shape[-1], layers=config["fusion_layers"],
        heads=config["fusion_heads"], gate_init=config["context_gate_init"],
    ).to(device).eval()
    fusion.load_state_dict(state, strict=True)
    del checkpoint

    corrected = torch.empty_like(original, device="cpu")
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.inference_mode():
            for start in range(0, len(original), batch_size):
                stop = min(start + batch_size, len(original))
                features = fusion(original[start:stop].to(device=device, dtype=torch.float32))
                if not torch.isfinite(features).all():
                    raise RuntimeError(f"Fusion still produces non-finite values at rows {start}:{stop}")
                corrected[start:stop] = features.to(device="cpu", dtype=original.dtype)
                print(f"Repaired {stop:,}/{len(original):,} panoramas", end="\r", flush=True)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    print()

    index["embeddings"] = corrected
    index["repair_note"] = "Cross-view embeddings regenerated from raw_embeddings with disabled PyTorch MHA eval fastpath"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Commit a complete new index atomically, never replacing the old index.
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        torch.save(index, temporary)
        if output.exists():
            raise FileExistsError(f"Output appeared during repair: {output}")
        os.link(temporary, output)  # Refuse to replace an existing destination.
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Saved finite embeddings to {output}; original index left unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    args = parser.parse_args()
    repair(args.checkpoint, args.index, args.output, args.batch_size, args.device)


if __name__ == "__main__":
    main()
