#!/usr/bin/env python3

import argparse
import os
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def convert_zip_to_tar(zip_path: Path) -> None:
    tar_path = zip_path.with_suffix(".tar")
    temp_path = tar_path.with_suffix(".tar.tmp")

    if tar_path.exists():
        return

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            with tarfile.open(temp_path, "w") as tar_file:
                for info in zip_file.infolist():
                    if info.is_dir():
                        continue

                    # Prevent path traversal such as ../../file
                    member_path = Path(info.filename)

                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise ValueError(
                            f"Unsafe path in {zip_path}: {info.filename}"
                        )

                    file_obj = zip_file.open(info)

                    tar_info = tarfile.TarInfo(info.filename)
                    tar_info.size = info.file_size
                    tar_info.mtime = 0o0

                    tar_file.addfile(tar_info, file_obj)

        # Atomic rename: incomplete conversions never appear as valid .tar files.
        os.replace(temp_path, tar_path)

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert all ZIP files in a folder to TAR files."
    )

    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of concurrent conversions.",
    )

    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()

    zip_files = sorted(folder.glob("*.zip"))

    if not zip_files:
        print(f"No ZIP files found in {folder}")
        return

    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_zip_to_tar, path): path
            for path in zip_files
        }

        with tqdm(
            total=len(futures),
            desc="Converting",
            unit="archive",
        ) as progress:
            for future in as_completed(futures):
                path = futures[future]

                try:
                    future.result()
                except Exception as exc:
                    errors.append((path, exc))

                progress.update(1)

    if errors:
        print("\nErrors:")
        for path, error in errors:
            print(f"  {path.name}: {error}")

        raise SystemExit(1)

    print(f"\nConverted {len(zip_files)} archives.")


if __name__ == "__main__":
    main()