"""Create a report-ready world map of panorama density from dataset metadata.

Example:
    python tools/plot_dataset_density.py \
        --dataset /home/austen/Street-View-Harvester/datasets/world

Only metadata.csv is read; image archives are never opened. The optional land
outline uses the same Natural Earth GeoJSON as tools/build_grid.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator


def sample_locations(metadata_path: Path, fraction: float, maximum: int, seed: int):
    """Stream the CSV and uniformly retain at most `maximum` selected panoramas."""
    rng = random.Random(seed)
    sample: list[tuple[float, float]] = []
    valid = selected = skipped = 0

    with metadata_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if {"pano_lat", "pano_lon"} <= fields:
            lat_col, lon_col = "pano_lat", "pano_lon"
        elif {"lat", "lon"} <= fields:
            lat_col, lon_col = "lat", "lon"
        else:
            raise ValueError("Metadata needs pano_lat/pano_lon or lat/lon columns")

        for row in reader:
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            except (ValueError, TypeError):
                skipped += 1
                continue
            if not np.isfinite(lat) or not np.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                skipped += 1
                continue

            valid += 1
            if rng.random() >= fraction:
                continue

            selected += 1
            point = (lon, lat)
            if len(sample) < maximum:
                sample.append(point)
            else:
                # Reservoir sampling avoids bias toward early metadata rows.
                index = rng.randrange(selected)
                if index < maximum:
                    sample[index] = point

    if not sample:
        raise ValueError("No panorama locations selected; increase --fraction or check metadata")
    return np.asarray(sample, dtype=np.float64), valid, selected, skipped


def land_geometry(path: Path, stride: int):
    """Load simplified outlines and fills from a GeoJSON Polygon/MultiPolygon."""
    with path.open(encoding="utf-8") as stream:
        geojson = json.load(stream)

    if geojson.get("type") == "FeatureCollection":
        geometries = (feature.get("geometry") for feature in geojson.get("features", []))
    elif geojson.get("type") == "Feature":
        geometries = [geojson.get("geometry")]
    else:
        geometries = [geojson]

    polygons = []
    outlines = []
    for geometry in geometries:
        if not geometry:
            continue
        kind = geometry.get("type")
        if kind == "Polygon":
            parts = [geometry["coordinates"]]
        elif kind == "MultiPolygon":
            parts = geometry["coordinates"]
        else:
            continue

        for polygon in parts:
            if not polygon:
                continue
            exterior = np.asarray(polygon[0], dtype=float)
            if len(exterior) < 3:
                continue
            # Reduces the complexity of the vector SVG when using 10m coastlines.
            exterior = np.vstack((exterior[::stride], exterior[-1]))
            if len(exterior) < 3:
                continue
            polygons.append(exterior)

            # Split outlines at the date line to avoid lines across the map.
            breaks = np.flatnonzero(np.abs(np.diff(exterior[:, 0])) > 180) + 1
            for segment in np.split(exterior, breaks):
                if len(segment) >= 2:
                    outlines.append(segment)
    return polygons, outlines


def plot_density(points: np.ndarray, valid: int, output_dir: Path, grid: int,
                 land_path: Path | None, coastline_stride: int, dpi: int):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "svg.fonttype": "none",  # Retain editable vector text.
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    fig.subplots_adjust(left=0.025, right=0.87, bottom=0.035, top=0.965)
    ax.set_facecolor("#f8f8f8")

    if land_path and land_path.is_file():
        polygons, outlines = land_geometry(land_path, coastline_stride)
        if polygons:
            ax.add_collection(PolyCollection(polygons, facecolors="#eeeeee", edgecolors="none", zorder=0))
            ax.add_collection(LineCollection(outlines, colors="#999999", linewidths=0.32, zorder=3))
    elif land_path:
        print(f"Note: land boundary not found ({land_path}); rendering density without coastlines")

    # Adjust the sampled counts back to estimated full-dataset counts.
    # Using the observed sampling fraction also handles --max-samples caps.
    scale = valid / len(points)
    print(f"Scaling sample density by {scale:.2f} to estimate full-dataset counts")
    density = ax.hexbin(
        points[:, 0], points[:, 1],
        C=np.full(len(points), scale),
        reduce_C_function=np.sum,
        gridsize=(grid, max(1, grid // 2)),
        extent=(-180, 180, -90, 90),
        mincnt=1,
        norm=LogNorm(),
        cmap="YlOrRd",
        linewidths=0,
        zorder=2,
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(density, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Estimated panoramas per hexagon", fontsize=9)
    colorbar.outline.set_linewidth(0.5)
    colorbar.locator = LogLocator(base=10, subs=(1, 2, 5))
    colorbar.formatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    colorbar.update_ticks()

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        destination = output_dir / f"panorama_density.{extension}"
        fig.savefig(destination, dpi=dpi)
        print(f"Saved {destination}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset directory containing metadata.csv (or path directly to a CSV)")
    parser.add_argument("--fraction", type=float, default=0.10,
                        help="Fraction of panorama rows to randomly sample (default: 0.10)")
    parser.add_argument("--max-samples", type=int, default=100_000,
                        help="Maximum number of locations to plot (default: 100000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid", type=int, default=180,
                        help="Hexagon density resolution across the map (default: 180)")
    parser.add_argument("--land-geojson", type=Path,
                        default=Path("outputs/grids/ne_10m_land.geojson"),
                        help="Optional Natural Earth land polygons for coastlines")
    parser.add_argument("--coastline-stride", type=int, default=12,
                        help="Keep every Nth coastline vertex to limit SVG size")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/figures/dataset_density"))
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    if not (0 < args.fraction <= 1.00):
        parser.error("--fraction must be in (0, 0.10] for a sample of at most 10%")
    if args.max_samples < 1 or args.grid < 1 or args.coastline_stride < 1 or args.dpi < 1:
        parser.error("Sample limit, grid, coastline stride, and DPI must be positive")

    metadata = args.dataset / "metadata.csv" if args.dataset.is_dir() else args.dataset
    if not metadata.is_file():
        parser.error(f"Metadata CSV not found: {metadata}")

    points, valid, selected, skipped = sample_locations(
        metadata, args.fraction, args.max_samples, args.seed)
    print(f"Valid panorama locations: {valid:,}; invalid: {skipped:,}")
    print(f"Sampled: {len(points):,} (initially selected {selected:,} before cap)")
    plot_density(points, valid, args.output_dir, args.grid,
                 args.land_geojson, args.coastline_stride, args.dpi)


if __name__ == "__main__":
    main()
