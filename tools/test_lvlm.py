from modules.lvlm import QwenVLBackend
from modules.dataset import PanoramaDataset
from modules.panorama import split_panorama
import matplotlib.pyplot as plt


dataset = PanoramaDataset(
    "test_dataset",
    shuffle=False,
    exclude_columns=[
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
    ],
)

image, metadata = next(iter(dataset))

subimages = split_panorama(image)

images = [crop.image for crop in subimages]

backend = QwenVLBackend(
    model_name="cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.75,
    max_model_len=1024,
    dtype="auto",
)

responses = backend.generate_batch(
    images=images,
    prompt="Describe this image",
    max_tokens=128,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.0,
)

for crop, response in zip(subimages, responses):
    print(f"\n--- Subimage {crop.subimage_index} ---")
    print(response.text)

    plt.imshow(crop.image)
    plt.axis("off")
    plt.show()