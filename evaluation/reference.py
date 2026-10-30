"""Build and load the training image reference database."""

from pathlib import Path
import torch
from tqdm import tqdm
from modules.dataset import PanoramaDataset
from evaluation.inference import encode_image


@torch.inference_mode()
def build_reference_index(model, processor, dataset_path, device, output_path, max_images=None):
    """Encode training images and store their embeddings with coordinates."""
    dataset = PanoramaDataset(dataset_path, shuffle=False)
    loader = dataset.loader(batch_size=1, num_workers=0)

    embeddings = []
    records = []

    progress = tqdm(loader, desc="Building reference index", total=max_images)

    for images, metadata_batch in progress:
        image = images[0]
        metadata = metadata_batch[0]

        try:
            lat = metadata.get("lat", metadata.get("pano_lat"))
            lon = metadata.get("lon", metadata.get("pano_lon"))
            if lat is None or lon is None:
                raise KeyError(f"No lat/lon in metadata columns: {list(metadata)}")

            embedding = encode_image(model, processor, image, device)
            embeddings.append(embedding.cpu())

            records.append({
                "image_id": metadata.get("panoid"),
                "image_path": metadata.get("image_path"),
                "latitude": float(lat),
                "longitude": float(lon),
            })
        finally:
            image.close()

        if max_images is not None and len(records) >= max_images:
            break

    if not embeddings:
        raise RuntimeError("Reference dataset contains no images")

    index = {
        "embeddings": torch.cat(embeddings),
        "records": records,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, output_path)

    return index


def load_reference_index(path):
    """Load previously cached training image embeddings."""
    return torch.load(path, map_location="cpu", weights_only=True)