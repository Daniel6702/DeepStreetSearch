"""Load the exact BLIP/LoRA encoder used by LaVPR's cross-modal evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_MODEL = "Salesforce/blip-itm-base-coco"
BLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_model(lavpr_path: str | Path, adapter_path: str | Path, device: str):
    """Return (BLIP with trained LaVPR LoRA, BLIP processor, image transform)."""
    import torch
    import torchvision.transforms as transforms
    from peft import PeftModel
    from transformers import BlipProcessor

    lavpr_path = Path(lavpr_path).resolve()
    if not (lavpr_path / "model" / "blip_model.py").is_file():
        raise FileNotFoundError(f"LaVPR submodule unavailable: {lavpr_path}")
    if not (Path(adapter_path) / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"LaVPR LoRA weights unavailable: {adapter_path}")

    # Load the exact LaVPR BLIP implementation without touching sys.path or
    # importing its heavyweight training dependencies.
    source = lavpr_path / "model" / "blip_model.py"
    spec = importlib.util.spec_from_file_location("lavpr_blip_model", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import LaVPR BLIP implementation: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    BlipForImageTextRetrievalWrapper = module.BlipForImageTextRetrievalWrapper

    base = BlipForImageTextRetrievalWrapper.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    model = model.to(device).eval()
    processor = BlipProcessor.from_pretrained(BASE_MODEL)

    # Keep the same transform ordering and 384px resolution as LaVPR's TestDataset.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=BLIP_MEAN, std=BLIP_STD),
        transforms.Resize((384, 384), antialias=True),
    ])
    return model, processor, transform


def encode_images(model, transform, images: list, device: str):
    """Six perspective views in, six normalized 256D image vectors out."""
    import torch
    import torch.nn.functional as F

    pixels = torch.stack([transform(image) for image in images]).to(device)
    with torch.inference_mode():
        vectors = model.encode_image(pixels)[:, 0, :]
    return F.normalize(vectors.float(), dim=-1).cpu()


def encode_texts(model, processor, texts: list[str], device: str):
    """Match LaVPR's evaluation path: BLIP tokens -> trained text CLS vectors."""
    import torch
    import torch.nn.functional as F

    tokens = processor(text=texts, padding=True, return_tensors="pt")
    lengths = tokens["attention_mask"].sum(dim=1).tolist()
    with torch.inference_mode():
        vectors = model.encode_text(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
        )[:, 0, :]
    return F.normalize(vectors.float(), dim=-1).cpu(), lengths
