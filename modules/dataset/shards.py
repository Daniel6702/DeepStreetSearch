from __future__ import annotations

import random
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO, TypeVar

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_SHARD = "__source_shard__"

T = TypeVar("T")


def image_key(filename: str) -> str:
    key = Path(filename).stem
    if key.endswith("_360"):
        key = key[:-4]
    return key


def decode_image(fileobj: BinaryIO) -> Image.Image:
    """Fully decode an image and detach it from the archive stream."""
    with Image.open(fileobj) as image:
        image.load()
        if image.mode != "RGB":
            return image.convert("RGB")
        return image.copy()


def find_shards(dataset_root: str | Path) -> list[str]:
    images_dir = Path(dataset_root) / "images"
    shards = [*images_dir.glob("shard_*.tar"), *images_dir.glob("shard_*.zip")]
    return sorted(str(path) for path in shards)


def _buffer_shuffle(iterator: Iterator[T], buffer_size: int, rng: random.Random) -> Iterator[T]:
    """Shuffle lightweight archive references without buffering decoded images."""
    if buffer_size <= 1:
        yield from iterator
        return

    buffer: list[T] = []
    for item in iterator:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue

        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item

    rng.shuffle(buffer)
    yield from buffer


def _iter_tar(shard_path: str, shuffle_buffer: int, rng: random.Random | None) -> Iterator[tuple[str, Image.Image]]:
    with tarfile.open(shard_path, mode="r:*") as archive:
        members = (
            member
            for member in archive
            if member.isfile() and Path(member.name).suffix.lower() in IMAGE_EXTENSIONS
        )
        if rng is not None:
            members = _buffer_shuffle(members, shuffle_buffer, rng)

        for member in members:
            fileobj = archive.extractfile(member)
            if fileobj is None:
                continue

            with fileobj:
                image = decode_image(fileobj)

            yield member.name, image


def _iter_zip(shard_path: str, shuffle_buffer: int, rng: random.Random | None) -> Iterator[tuple[str, Image.Image]]:
    with zipfile.ZipFile(shard_path, mode="r") as archive:
        members = (
            member
            for member in archive.infolist()
            if not member.is_dir() and Path(member.filename).suffix.lower() in IMAGE_EXTENSIONS
        )
        if rng is not None:
            members = _buffer_shuffle(members, shuffle_buffer, rng)

        for member in members:
            with archive.open(member, mode="r") as fileobj:
                image = decode_image(fileobj)

            yield member.filename, image


def iter_shard_images(
    shard_path: str,
    shuffle_buffer: int = 1,
    rng: random.Random | None = None,
) -> Iterator[tuple[str, Image.Image]]:
    suffix = Path(shard_path).suffix.lower()
    if suffix == ".tar":
        yield from _iter_tar(shard_path, shuffle_buffer, rng)
    elif suffix == ".zip":
        yield from _iter_zip(shard_path, shuffle_buffer, rng)
    else:
        raise ValueError(f"Unsupported shard format: {shard_path}")
