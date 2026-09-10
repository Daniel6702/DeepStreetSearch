import cv2


def split_panorama(
    image_path: str,
    output_prefix: str = "square",
    tile_count: int = 6,
    tile_size: int = 768,
) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    height, width = img.shape[:2]
    tile_width = width // tile_count

    for index in range(tile_count):
        start_x = index * tile_width
        end_x = width if index == tile_count - 1 else (index + 1) * tile_width
        tile = img[:, start_x:end_x]
        square = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

        output_path = f"{output_prefix}_{index + 1}.jpg"
        if not cv2.imwrite(output_path, square):
            raise OSError(f"Unable to write image: {output_path}")

split_panorama("image.jpg")