from adaptive_geogrid import load_grid
import matplotlib.pyplot as plt
from modules.dataset import PanoramaDataset
import numpy as np

grid = load_grid("grids/world_full.aggrid")

metadata = PanoramaDataset.load_metadata("/home/austen/Street-View-Harvester/datasets/world")

points = []
seen = set()

for row in metadata.values():
    lat = row.get("lat", row.get("pano_lat"))
    lon = row.get("lon", row.get("pano_lon"))

    if lat is None or lon is None:
        raise KeyError("Metadata must contain lat/lon or pano_lat/pano_lon")

    key = row.get("panoid") or row.get("pano_id") or (lon, lat)
    if key in seen:
        continue

    seen.add(key)
    points.append((float(lon), float(lat)))

points = np.asarray(points, dtype=float)
points = points[:10000]
print(f"Building grid from {len(points):,} unique panorama locations")

ax = grid.plot(
    levels=[0,1,2],
    points=points,
    point_sample=10_000,
    linewidth=0.1,
    boundary_linewidth=0.25,
    hierarchy_lane_alpha=0.82,
    hierarchy_lane_gap=0.0,
    point_size=0.05,
    point_alpha=0.20,
)

ax.set_title("Grid Visual")
plt.tight_layout()

plt.savefig("grid_visual.svg", bbox_inches="tight")
plt.close()

print("Saved grid_visual.svg")