"""Generate descriptions of unseen panoramas using Qwen3-VL."""

from PIL import Image

from config import (
    MODEL,
    MAX_TOKENS,
    MAX_MODEL_LENGTH,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
    PROMPT1,
)

from modules.panorama import split_panorama
from modules.lvlm import QwenVLBackend 


class PanoramaDescriber:
    def __init__(self):
        """Load the same VLM used to generate training descriptions."""
        self.backend = QwenVLBackend(
            model_name=MODEL,
            max_model_len=MAX_MODEL_LENGTH,
        )

    def describe(self, image_path, prompt=PROMPT1):
        """Generate descriptions for all panorama views."""
        with Image.open(image_path) as image:
            panorama = image.convert("RGB")

        crops = split_panorama(panorama)

        try:
            # Describe all views in one batch.
            responses = self.backend.generate_batch(
                images=[crop.image for crop in crops],
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
            )

            descriptions = [
                response.text
                for response in responses
            ]

            # Combine view descriptions into one panorama description.
            return " ".join(descriptions)

        finally:
            for crop in crops:
                crop.image.close()
            panorama.close()