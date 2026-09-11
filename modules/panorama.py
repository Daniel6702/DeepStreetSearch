from __future__ import annotations

from dataclasses import dataclass
from typing import List

from PIL import Image


NUM_VIEWS = 6
CROP_SIZE = 1024
OUTPUT_SIZE = 768


@dataclass
class PanoramaCrop:
    subimage_index: int
    image: Image.Image


def _circular_crop_x(
    image: Image.Image,
    left_px: int,
    crop_size: int,
) -> Image.Image:
    """Crop horizontally with wrap-around at the panorama boundary."""
    width, height = image.size
    left_px %= width

    if left_px + crop_size <= width:
        return image.crop((left_px, 0, left_px + crop_size, height))

    right = image.crop((left_px, 0, width, height))
    remaining_width = crop_size - right.width
    left = image.crop((0, 0, remaining_width, height))

    output = Image.new("RGB", (crop_size, height))
    output.paste(right, (0, 0))
    output.paste(left, (right.width, 0))

    return output


def get_panorama_subimage(
    panorama: Image.Image,
    index: int,
) -> Image.Image:
    """
    Return a single panorama subimage.

    For a 4096x1024 panorama, the six views are centered at:
    0, 60, 120, 180, 240, and 300 degrees.
    """
    width, height = panorama.size

    if height != CROP_SIZE:
        raise ValueError(
            f"Expected panorama height {CROP_SIZE}, got {height}."
        )

    if not 0 <= index < NUM_VIEWS:
        raise ValueError(
            f"Subimage index must be between 0 and {NUM_VIEWS - 1}, got {index}."
        )

    step = width / NUM_VIEWS
    center_x = index * step

    left_px = int(round(center_x - CROP_SIZE / 2)) % width

    crop = _circular_crop_x(
        panorama,
        left_px,
        CROP_SIZE,
    )

    if OUTPUT_SIZE != CROP_SIZE:
        crop = crop.resize(
            (OUTPUT_SIZE, OUTPUT_SIZE),
            Image.Resampling.LANCZOS,
        )

    return crop


def split_panorama(
    panorama: Image.Image,
) -> List[PanoramaCrop]:
    """Split the panorama into all predefined subimages."""

    return [
        PanoramaCrop(
            subimage_index=index,
            image=get_panorama_subimage(panorama, index),
        )
        for index in range(NUM_VIEWS)
    ]