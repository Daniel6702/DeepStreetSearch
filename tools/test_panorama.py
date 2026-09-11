import matplotlib.pyplot as plt

from modules.dataset import PanoramaDataset
from modules.panorama import split_panorama, PanoramaCrop

dataset = PanoramaDataset("test_dataset", 
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
                            "fov"
                            ],
                        )

image, metadata = next(iter(dataset))

subimages = split_panorama(image)

print(subimages[0].subimage_index)
print(subimages[0].image.height)

plt.imshow(subimages[0].image)
plt.axis("off")
plt.show()