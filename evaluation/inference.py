"""Load the trained model and encode images or text."""

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from adaptive_geogrid import load_grid
from modules.model import GeoSearchModel
from modules.panorama import split_panorama


def load_model(checkpoint, grid_path, model_name, device):
    """Load the trained GeoSearchModel and its processor."""
    grid = load_grid(grid_path)

    model = GeoSearchModel(grid, model_name=model_name)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])

    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


@torch.inference_mode()
def encode_image(model, processor, image, device):
    """Encode a panorama into a single normalized embedding."""
    crops = split_panorama(image)

    try:
        if not crops:
            raise ValueError("Panorama produced no crops")

        inputs = processor(
            images=[crop.image for crop in crops],
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        _, embeddings = model.encode_images(
            inputs, batch_size=1, num_views=len(crops)
        )

        # Average the view embeddings into one panorama embedding.
        embedding = embeddings.mean(dim=0, keepdim=True)
        return F.normalize(embedding.float(), dim=-1)

    finally:
        for crop in crops:
            crop.image.close()


@torch.inference_mode()
def encode_text(model, processor, descriptions, device):
    """Encode one or more descriptions using the trained SigLIP text encoder."""
    if isinstance(descriptions, str):
        descriptions = [descriptions]

    inputs = processor(
        text=descriptions,
        padding=True,
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask"}
    }

    return model.encode_text(inputs).float()