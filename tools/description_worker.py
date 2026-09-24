from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import sys
import time
from typing import Sequence

from tqdm import tqdm

from config import (
    CROP_SIZE,
    DATASET_PATH,
    MAX_MODEL_LENGTH,
    MAX_TOKENS,
    MODEL,
    NUM_VIEWS,
    OUTPUT_SIZE,
    PROMPT1,
    PROMPT2,
    PROMPT3,
    PROMPT4,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_P,
)
from modules.dataset import PanoramaDataset
from modules.descriptions import DescriptionStore


PROMPTS = [PROMPT1, PROMPT2, PROMPT3, PROMPT4]

# Changing this deliberately gives the new storage layout a different run
# signature from older smoke tests that wrote to one global SQLite database.
STORAGE_VERSION = "per-shard-sqlite-v1"


def _select_prompts(prompt_numbers: Sequence[int]) -> list[tuple[int, str]]:
    """Return stable zero-based database indices paired with prompt text."""
    unique_numbers = sorted(set(prompt_numbers))
    if not unique_numbers:
        raise ValueError("At least one prompt must be selected")

    for number in unique_numbers:
        if not 1 <= number <= len(PROMPTS):
            raise ValueError(
                f"Prompt number must be between 1 and {len(PROMPTS)}, got {number}"
            )

    return [
        (number - 1, PROMPTS[number - 1])
        for number in unique_numbers
    ]


EXCLUDE_COLUMNS = [
    "pano_lat",
    "pano_lon",
    "pano_date",
    "query_lat",
    "query_lon",
    "snap_distance_m",
    "source",
    "source_value",
    "image_type",
    "yaw",
    "pitch",
    "fov",
    "image_path",
]


class StopRequested:
    def __init__(self) -> None:
        self.value = False

    def request(self, signum, frame) -> None:
        if not self.value:
            print(
                "\nStop requested. Finishing the current LVLM call and "
                "saving its results before exiting...",
                flush=True,
            )
        self.value = True


class ShardClaimer:
    """Filesystem-based shard claiming with stale-claim recovery."""

    def __init__(
        self,
        dataset_root: Path,
        stale_after_seconds: float,
        run_signature: str,
    ) -> None:
        self.root = dataset_root / ".description_worker" / "runs" / run_signature
        self.claims_dir = self.root / "claims"
        self.done_dir = self.root / "done"
        self.stale_after_seconds = stale_after_seconds

        self.claims_dir.mkdir(parents=True, exist_ok=True)
        self.done_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _name(shard: Path) -> str:
        return shard.name

    def _claim_path(self, shard: Path) -> Path:
        return self.claims_dir / f"{self._name(shard)}.claim"

    def _done_path(self, shard: Path) -> Path:
        return self.done_dir / f"{self._name(shard)}.done"

    def is_done(self, shard: Path) -> bool:
        return self._done_path(shard).exists()

    def try_claim(self, shard: Path) -> Path | None:
        if self.is_done(shard):
            return None

        claim_path = self._claim_path(shard)

        try:
            claim_path.mkdir()
        except FileExistsError:
            if not self._is_stale(claim_path):
                return None

            if not self._steal_stale_claim(claim_path):
                return None

            try:
                claim_path.mkdir()
            except FileExistsError:
                return None

        owner = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started": time.time(),
        }
        (claim_path / "owner.json").write_text(
            json.dumps(owner, indent=2),
            encoding="utf-8",
        )
        self.heartbeat(claim_path)
        return claim_path

    def _is_stale(self, claim_path: Path) -> bool:
        try:
            age = time.time() - claim_path.stat().st_mtime
        except FileNotFoundError:
            return False
        return age > self.stale_after_seconds

    def _steal_stale_claim(self, claim_path: Path) -> bool:
        tombstone = claim_path.with_name(
            f"{claim_path.name}.stale.{socket.gethostname()}.{os.getpid()}"
        )

        try:
            claim_path.rename(tombstone)
        except (FileNotFoundError, OSError):
            return False

        shutil.rmtree(tombstone, ignore_errors=True)
        return True

    @staticmethod
    def heartbeat(claim_path: Path) -> None:
        try:
            os.utime(claim_path, None)
        except FileNotFoundError:
            pass

    def mark_done(self, shard: Path, claim_path: Path) -> None:
        done_path = self._done_path(shard)
        temporary_path = done_path.with_name(
            f"{done_path.name}.tmp.{socket.gethostname()}.{os.getpid()}"
        )

        temporary_path.write_text(
            json.dumps(
                {
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "finished": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(done_path)
        self.release(claim_path)

    @staticmethod
    def release(claim_path: Path | None) -> None:
        if claim_path is not None:
            shutil.rmtree(claim_path, ignore_errors=True)

    def claim_next(self, shards: Sequence[Path]) -> tuple[Path, Path] | None:
        candidates = list(shards)
        random.shuffle(candidates)

        for shard in candidates:
            claim_path = self.try_claim(shard)
            if claim_path is not None:
                return shard, claim_path

        return None


def _run_signature(selected_prompts: Sequence[tuple[int, str]]) -> str:
    payload = {
        "storage_version": STORAGE_VERSION,
        "model": MODEL,
        "prompts": [
            {"prompt_index": prompt_index, "text": prompt}
            for prompt_index, prompt in selected_prompts
        ],
        "max_model_length": MAX_MODEL_LENGTH,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "num_views": NUM_VIEWS,
        "crop_size": CROP_SIZE,
        "output_size": OUTPUT_SIZE,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LVLM descriptions for panorama dataset shards."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(DATASET_PATH),
        help="Dataset root containing metadata.csv and images/shard_*.tar|zip.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help=(
            "Optional single SQLite database. Intended for single-worker/debug "
            "runs. By default, each shard gets its own database under "
            "<dataset>/descriptions/ to avoid concurrent SQLite writes."
        ),
    )
    parser.add_argument(
        "--prompts",
        type=int,
        nargs="+",
        default=list(range(1, len(PROMPTS) + 1)),
        metavar="N",
        help=(
            "Prompt numbers to run (1-based). Example: --prompts 1 2 now, "
            "then --prompts 3 4 later. Stored prompt_index values remain "
            "stable at 0..3."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help=(
            "GPU visibility override. Prefer setting CUDA_VISIBLE_DEVICES "
            "before starting Python in distributed/orchestrated runs."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Panoramas loaded and processed together.",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=1,
        help=(
            "Loader worker processes. Each description worker handles one "
            "shard at a time, so values above 1 provide no benefit there."
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Prepared batches prefetched by each active loader worker.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--mm-processor-cache-gb",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--stale-after-minutes",
        type=float,
        default=120.0,
    )
    return parser.parse_args()


def _close_crops(crops_batch) -> None:
    for crops in crops_batch:
        for crop in crops:
            crop.image.close()


def process_batch(
    crops_batch,
    metadata_batch,
    backend,
    store: DescriptionStore,
    prompts: Sequence[tuple[int, str]],
    claimer: ShardClaimer,
    claim_path: Path,
    stop: StopRequested,
) -> int:
    panoids = [metadata["panoid"] for metadata in metadata_batch]
    existing = store.existing_keys(panoids)

    crops_by_panoid = {
        panoid: crops
        for panoid, crops in zip(panoids, crops_batch)
        if not all(
            (panoid, subimage_index, prompt_index) in existing
            for subimage_index in range(NUM_VIEWS)
            for prompt_index, _ in prompts
        )
    }

    generated = 0

    try:
        for prompt_index, prompt in prompts:
            request_images = []
            request_keys = []

            for panoid, crops in crops_by_panoid.items():
                for crop in crops:
                    key = (panoid, crop.subimage_index, prompt_index)
                    if key in existing:
                        continue

                    request_images.append(crop.image)
                    request_keys.append(key)

            if not request_images:
                continue

            responses = backend.generate_batch(
                images=request_images,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
            )

            if len(responses) != len(request_keys):
                raise RuntimeError(
                    f"LVLM returned {len(responses)} responses for "
                    f"{len(request_keys)} requests"
                )

            rows = [
                (panoid, subimage_index, prompt_index, response.text)
                for (panoid, subimage_index, prompt_index), response
                in zip(request_keys, responses)
            ]
            store.add_many(rows)
            existing.update(request_keys)
            generated += len(rows)
            claimer.heartbeat(claim_path)

            del responses, request_images, request_keys

            if stop.value:
                break

        return generated
    finally:
        _close_crops(crops_batch)


def process_shard(
    shard: Path,
    claim_path: Path,
    dataset_root: Path,
    metadata,
    backend,
    database_path: Path,
    claimer: ShardClaimer,
    stop: StopRequested,
    batch_size: int,
    loader_workers: int,
    prefetch_factor: int,
    prompts: Sequence[tuple[int, str]],
) -> bool:
    dataset = PanoramaDataset(
        dataset_root,
        shards=[shard],
        shuffle=False,
        exclude_columns=EXCLUDE_COLUMNS,
        metadata=metadata,
    )

    loader = dataset.loader(
        batch_size=batch_size,
        num_workers=loader_workers,
        prefetch_factor=prefetch_factor,
        prepare_panorama_crops=True,
    )

    progress = tqdm(
        loader,
        desc=shard.name,
        unit="batch",
        dynamic_ncols=True,
    )

    total_generated = 0

    try:
        with DescriptionStore(database_path) as store:
            for crops_batch, metadata_batch in progress:
                generated = process_batch(
                    crops_batch=crops_batch,
                    metadata_batch=metadata_batch,
                    backend=backend,
                    store=store,
                    prompts=prompts,
                    claimer=claimer,
                    claim_path=claim_path,
                    stop=stop,
                )
                total_generated += generated
                progress.set_postfix(generated=total_generated)
                claimer.heartbeat(claim_path)

                if stop.value:
                    return False
    finally:
        progress.close()

    return True


def main() -> int:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.loader_workers < 0:
        raise ValueError("--loader-workers cannot be negative")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be at least 1")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.mm_processor_cache_gb < 0:
        raise ValueError("--mm-processor-cache-gb cannot be negative")
    if args.stale_after_minutes <= 0:
        raise ValueError("--stale-after-minutes must be positive")

    # This remains useful for direct/manual runs. In the orchestrator,
    # CUDA_VISIBLE_DEVICES is set before Python starts instead.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from modules.lvlm import QwenVLBackend

    dataset_root = args.dataset.resolve()
    database_override = (
        args.database.resolve()
        if args.database is not None
        else None
    )
    database_dir = dataset_root / "descriptions"

    shards = [
        Path(path)
        for path in PanoramaDataset.find_shards(dataset_root)
    ]
    if not shards:
        raise FileNotFoundError(
            f"No shard_*.tar or shard_*.zip files found in "
            f"{dataset_root / 'images'}"
        )

    selected_prompts = _select_prompts(args.prompts)
    run_signature = _run_signature(selected_prompts)

    print(f"Dataset:   {dataset_root}")
    if database_override is None:
        print(f"Database:  per-shard SQLite files in {database_dir}")
    else:
        print(f"Database:  {database_override}")
    print(f"Shards:    {len(shards)}")
    print(f"Model:     {MODEL}")
    prompt_numbers = [prompt_index + 1 for prompt_index, _ in selected_prompts]
    print(f"Prompts:   {prompt_numbers} ({len(selected_prompts)} selected)")
    print(f"Batch:     {args.batch_size} panoramas")
    print(
        f"Loader:    {args.loader_workers} requested worker(s), "
        f"prefetch={args.prefetch_factor}"
    )
    print(f"MM cache:  {args.mm_processor_cache_gb:g} GiB")
    print(f"Run ID:    {run_signature}")

    metadata = PanoramaDataset.load_metadata(
        dataset_root,
        exclude_columns=EXCLUDE_COLUMNS,
    )

    stop = StopRequested()
    signal.signal(signal.SIGINT, stop.request)
    signal.signal(signal.SIGTERM, stop.request)

    max_model_len = None if MAX_MODEL_LENGTH == -1 else MAX_MODEL_LENGTH
    backend = QwenVLBackend(
        model_name=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        mm_processor_cache_gb=args.mm_processor_cache_gb,
    )

    claimer = ShardClaimer(
        dataset_root=dataset_root,
        stale_after_seconds=args.stale_after_minutes * 60.0,
        run_signature=run_signature,
    )

    current_claim: Path | None = None

    try:
        while not stop.value:
            claimed = claimer.claim_next(shards)

            if claimed is None:
                remaining = [
                    shard
                    for shard in shards
                    if not claimer.is_done(shard)
                ]

                if remaining:
                    print(
                        "No shard is currently claimable. The remaining "
                        "shards are owned by other workers."
                    )
                else:
                    print("All shards are complete.")
                break

            shard, current_claim = claimed
            print(f"\nClaimed {shard.name}", flush=True)

            database_path = (
                database_override
                if database_override is not None
                else database_dir / f"{shard.name}.sqlite"
            )

            try:
                completed = process_shard(
                    shard=shard,
                    claim_path=current_claim,
                    dataset_root=dataset_root,
                    metadata=metadata,
                    backend=backend,
                    database_path=database_path,
                    claimer=claimer,
                    stop=stop,
                    batch_size=args.batch_size,
                    loader_workers=args.loader_workers,
                    prefetch_factor=args.prefetch_factor,
                    prompts=selected_prompts,
                )
            except Exception:
                claimer.release(current_claim)
                current_claim = None
                raise

            if completed:
                claimer.mark_done(shard, current_claim)
                current_claim = None
                print(f"Completed {shard.name}", flush=True)
            else:
                claimer.release(current_claim)
                current_claim = None
                break

    finally:
        claimer.release(current_claim)

    return 0


if __name__ == "__main__":
    sys.exit(main())
