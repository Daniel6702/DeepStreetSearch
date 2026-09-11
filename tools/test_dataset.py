import matplotlib.pyplot as plt

from modules.dataset import PanoramaDataset

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

print(metadata)

plt.imshow(image)
plt.axis("off")
plt.show()