from __future__ import annotations

import argparse
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
    DATASET_PATH,
    MAX_MODEL_LENGTH,
    MAX_TOKENS,
    MODEL,
    PROMPT1,
    PROMPT2,
    PROMPT3,
    PROPMT4,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_P,
)
from modules.dataset import PanoramaDataset
from modules.descriptions import DescriptionStore
from modules.panorama import NUM_VIEWS, split_panorama


PROMPTS = [PROMPT1, PROMPT2, PROMPT3, PROPMT4]

# These metadata fields are not needed by the description worker. Keeping only
# panoid and the small set of useful fields saves a substantial amount of RAM
# on large datasets.
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
                "\nStop requested. Finishing the current LVLM batch and "
                "saving its results before exiting...",
                flush=True,
            )
        self.value = True


class ShardClaimer:
    """
    Minimal filesystem-based shard claiming.

    mkdir() is used as the atomic claim operation. Claims are heartbeated while
    work is progressing. If a worker dies without cleanup, another worker may
    reclaim the shard after stale_after_seconds.
    """

    def __init__(
        self,
        dataset_root: Path,
        stale_after_seconds: float,
        run_signature: str,
    ) -> None:
        self.root = dataset_root / ".description_worker"
        self.claims_dir = self.root / "claims"
        self.done_dir = self.root / "runs" / run_signature / "done"
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
        except FileNotFoundError:
            return False
        except OSError:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LVLM descriptions for panorama dataset shards."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(DATASET_PATH),
        help="Dataset root containing metadata.csv and images/shard_*.tar.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="SQLite output path. Default: <dataset>/descriptions.sqlite.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU visible to this worker, e.g. 0 or 1.",
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
        default=2,
        help="WebDataset loader worker processes.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--stale-after-minutes",
        type=float,
        default=120.0,
        help="Reclaim a shard if its worker has not heartbeated for this long.",
    )
    return parser.parse_args()


def process_batch(
    images,
    metadata_batch,
    backend,
    store: DescriptionStore,
    prompts: Sequence[str],
    claimer: ShardClaimer,
    claim_path: Path,
    stop: StopRequested,
) -> int:
    """
    Process one batch of panoramas.

    Existing database rows are skipped, so a shard can be resumed after a
    crash without repeating completed prompt/subimage work.
    """
    panoids = [metadata["panoid"] for metadata in metadata_batch]
    existing = store.existing_keys(panoids)

    crops_by_panoid = {}
    for image, panoid in zip(images, panoids):
        if all(
            (panoid, subimage_index, prompt_index) in existing
            for subimage_index in range(NUM_VIEWS)
            for prompt_index in range(len(prompts))
        ):
            continue

        crops_by_panoid[panoid] = split_panorama(image)

    generated = 0

    for prompt_index, prompt in enumerate(prompts):
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

        # Commit after every LVLM batch. A sudden crash therefore loses at most
        # the currently running generation call, not the entire shard.
        rows = [
            (panoid, subimage_index, prompt_index, response.text)
            for (panoid, subimage_index, prompt_index), response
            in zip(request_keys, responses)
        ]
        store.add_many(rows)
        existing.update(request_keys)
        generated += len(rows)
        claimer.heartbeat(claim_path)

        if stop.value:
            break

    return generated


def process_shard(
    shard: Path,
    claim_path: Path,
    dataset_root: Path,
    metadata,
    backend,
    store: DescriptionStore,
    claimer: ShardClaimer,
    stop: StopRequested,
    batch_size: int,
    loader_workers: int,
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
    )

    progress = tqdm(
        loader,
        desc=shard.name,
        unit="batch",
        dynamic_ncols=True,
    )

    for images, metadata_batch in progress:
        generated = process_batch(
            images=images,
            metadata_batch=metadata_batch,
            backend=backend,
            store=store,
            prompts=PROMPTS,
            claimer=claimer,
            claim_path=claim_path,
            stop=stop,
        )
        progress.set_postfix(generated=generated)
        claimer.heartbeat(claim_path)

        if stop.value:
            return False

    return True


def main() -> int:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.loader_workers < 0:
        raise ValueError("--loader-workers cannot be negative")
    if args.stale_after_minutes <= 0:
        raise ValueError("--stale-after-minutes must be positive")

    # Set this before importing vLLM/torch through modules.lvlm.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from modules.lvlm import QwenVLBackend

    dataset_root = args.dataset.resolve()
    database_path = (
        args.database.resolve()
        if args.database is not None
        else dataset_root / "descriptions.sqlite"
    )

    shards = [
        Path(path)
        for path in PanoramaDataset.find_shards(dataset_root)
    ]
    if not shards:
        raise FileNotFoundError(
            f"No shard_*.tar files found in {dataset_root / 'images'}"
        )

    print(f"Dataset:  {dataset_root}")
    print(f"Database: {database_path}")
    print(f"Shards:   {len(shards)}")
    print(f"Model:    {MODEL}")
    print(f"Prompts:  {len(PROMPTS)}")
    print(f"Batch:    {args.batch_size} panoramas")

    # metadata.csv is loaded once and then reused for every claimed shard.
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
    )

    current_claim: Path | None = None

    with DescriptionStore(database_path) as store:
        pass

    claimer = ShardClaimer(
        dataset_root=dataset_root,
        stale_after_seconds=args.stale_after_minutes * 60.0,
    )

    try:
        with DescriptionStore(database_path) as store:
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

                try:
                    completed = process_shard(
                        shard=shard,
                        claim_path=current_claim,
                        dataset_root=dataset_root,
                        metadata=metadata,
                        backend=backend,
                        store=store,
                        claimer=claimer,
                        stop=stop,
                        batch_size=args.batch_size,
                        loader_workers=args.loader_workers,
                    )
                except Exception:
                    # Release on ordinary Python failures so another worker can
                    # retry immediately. Hard kills/OOMs leave the claim behind;
                    # those are recovered using the stale-claim timeout.
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
        # SIGINT/SIGTERM reaches here after the current saved LVLM batch.
        # A SIGKILL/OOM cannot run cleanup, which is why stale claims exist.
        claimer.release(current_claim)

    return 0


if __name__ == "__main__":
    sys.exit(main())
