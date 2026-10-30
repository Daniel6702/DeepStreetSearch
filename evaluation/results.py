"""Save evaluation results as JSON and CSV."""

import csv
import json
from pathlib import Path


def save_results(matches, image_path, description, latitude, longitude, output_dir):
    """Save the predicted location and top retrieved images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "image": str(image_path),
        "description": description,
        "ground_truth": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "prediction": matches[0],
        "matches": matches,
    }

    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with (output_dir / "matches.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank", "image_id", "image_path", "latitude",
                "longitude", "similarity", "distance_km",
            ],
        )
        writer.writeheader()
        writer.writerows(matches)