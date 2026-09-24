from __future__ import annotations

import argparse

import numpy as np

from adaptive_geogrid import tessellate
from modules.dataset import PanoramaDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset root containing metadata.csv", default="/home/austen/Street-View-Harvester/datasets/world")
    parser.add_argument("--boundary", help="GeoJSON boundary used for the grid", default="grids/ne_10m_land.geojson")
    parser.add_argument("--output", default="grids/world_full.aggrid", help="Output .aggrid file")
    parser.add_argument("--target-points", type=int, default=200, help="Approximate panoramas per fine cell")
    parser.add_argument("--levels", type=int, default=5, help="Number of coarser hierarchy levels")
    parser.add_argument("--branching-factor", type=int, default=3, help="Children per parent level")
    parser.add_argument("--mode", default="geodesic", choices=["power", "warped", "geodesic", "graph"])
    parser.add_argument("--projected-crs", default="EPSG:8857", help="Projected CRS, e.g. EPSG:8857 for worldwide grids")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = PanoramaDataset.load_metadata(args.dataset)

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
    points = points[:1_000_000]
    print(f"Building grid from {len(points):,} unique panorama locations")
    #geodesic_grid_size
    grid = tessellate(
        points=points,
        boundary=args.boundary,
        target_points=args.target_points,
        mode=args.mode,
        geodesic_grid_size=8192,
        projected_crs=args.projected_crs,
        verbose=True,
    )

    grid.build_hierarchy(
        branching_factor=args.branching_factor,
        levels=args.levels,
    )

    grid.save(args.output)

    print(f"Saved {args.output}")
    print(f"Classes fine -> coarse: {grid.n_classes}")


if __name__ == "__main__":
    main()