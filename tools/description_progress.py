from __future__ import annotations

import argparse
import csv
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path


NUM_VIEWS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor aggregate description-generation progress."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset root.",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Prompt numbers being generated. Default: 1 2",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between updates. Default: 10",
    )
    return parser.parse_args()


def count_panoramas(metadata_path: Path) -> int:
    with metadata_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.reader(f)

        try:
            next(reader)
        except StopIteration:
            return 0

        return sum(1 for _ in reader)


def count_descriptions(
    database_dir: Path,
    prompt_indices: list[int],
) -> tuple[int, int]:
    total = 0
    databases = 0

    placeholders = ",".join(
        "?" for _ in prompt_indices
    )

    query = f"""
        SELECT COUNT(*)
        FROM descriptions
        WHERE prompt_index IN ({placeholders})
    """

    for database in database_dir.glob("shard_*.sqlite"):
        try:
            connection = sqlite3.connect(
                f"file:{database.resolve()}?mode=ro",
                uri=True,
                timeout=1.0,
            )

            try:
                row = connection.execute(
                    query,
                    prompt_indices,
                ).fetchone()

                total += row[0]
                databases += 1

            finally:
                connection.close()

        except sqlite3.Error:
            # A worker may be committing to the database at this exact
            # moment. Skip it for this sample; it will be counted next time.
            continue

    return total, databases


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))

    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"

    if hours:
        return f"{hours}h {minutes:02d}m"

    if minutes:
        return f"{minutes}m {seconds:02d}s"

    return f"{seconds}s"


def main() -> int:
    args = parse_args()

    dataset_root = args.dataset.resolve()
    metadata_path = dataset_root / "metadata.csv"
    database_dir = dataset_root / "descriptions"

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    prompt_numbers = sorted(set(args.prompts))

    if any(prompt < 1 for prompt in prompt_numbers):
        raise ValueError("Prompt numbers must be >= 1")

    prompt_indices = [
        prompt - 1
        for prompt in prompt_numbers
    ]

    print("Counting panoramas...", flush=True)

    panoramas = count_panoramas(metadata_path)

    expected = (
        panoramas
        * NUM_VIEWS
        * len(prompt_indices)
    )

    descriptions_per_panorama = (
        NUM_VIEWS
        * len(prompt_indices)
    )

    print(f"Dataset:     {dataset_root}")
    print(f"Panoramas:   {panoramas:,}")
    print(f"Prompts:     {prompt_numbers}")
    print(f"Expected:    {expected:,} descriptions")
    print(f"Interval:    {args.interval:g}s")
    print()

    previous_count = None
    previous_time = None
    smoothed_rate = None

    try:
        while True:
            now_monotonic = time.monotonic()

            generated, databases = count_descriptions(
                database_dir,
                prompt_indices,
            )

            progress = (
                generated / expected
                if expected
                else 0.0
            )

            rate = None

            if (
                previous_count is not None
                and previous_time is not None
                and generated >= previous_count
            ):
                elapsed = now_monotonic - previous_time

                if elapsed > 0:
                    rate = (
                        generated - previous_count
                    ) / elapsed

                    if rate > 0:
                        if smoothed_rate is None:
                            smoothed_rate = rate
                        else:
                            # Exponential smoothing prevents the ETA from
                            # jumping wildly between samples.
                            smoothed_rate = (
                                0.25 * rate
                                + 0.75 * smoothed_rate
                            )

            remaining = max(
                0,
                expected - generated,
            )

            print("\033[2J\033[H", end="")

            print("Description generation")
            print("======================")
            print(
                f"Progress:       "
                f"{generated:,} / {expected:,} "
                f"({progress * 100:.2f}%)"
            )
            print(
                f"Panoramas eq.:  "
                f"{generated / descriptions_per_panorama:,.1f} "
                f"/ {panoramas:,}"
            )
            print(
                f"Shard DBs:      "
                f"{databases:,}"
            )

            if smoothed_rate is not None and smoothed_rate > 0:
                panorama_rate = (
                    smoothed_rate
                    / descriptions_per_panorama
                )

                eta_seconds = remaining / smoothed_rate

                finish_time = (
                    datetime.now().astimezone()
                    + timedelta(seconds=eta_seconds)
                )

                print(
                    f"Rate:           "
                    f"{smoothed_rate:,.1f} descriptions/s"
                )
                print(
                    f"                "
                    f"{panorama_rate:,.2f} panoramas/s"
                )
                print(
                    f"Remaining:      "
                    f"{format_duration(eta_seconds)}"
                )
                print(
                    f"Estimated done: "
                    f"{finish_time:%Y-%m-%d %H:%M:%S %Z}"
                )
            else:
                print("Rate:           measuring...")
                print("Remaining:      measuring...")
                print("Estimated done: measuring...")

            print(
                f"Updated:         "
                f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S}"
            )

            if generated >= expected:
                print()
                print("COMPLETE")
                break

            previous_count = generated
            previous_time = now_monotonic

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped progress monitor.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())