"""Offline checks for retrieval example selection and preview lookup."""

import hashlib
import csv
import json
import tempfile
import unittest
from pathlib import Path

from text_location_eval.inspect_results import choose_queries, inspect, preview_path


class InspectResultsTests(unittest.TestCase):
    def setUp(self):
        self.retrieval = {"queries": [{"id": f"Q{n:03d}"} for n in range(1, 7)]}
        self.metrics = {f"Q{n:03d}": {"mean_relevance@10": str(n - 1)}
                        for n in range(1, 7)}

    def test_representative_selection_includes_extremes(self):
        selected = choose_queries(self.retrieval, self.metrics, None, 3)
        self.assertEqual([row["id"] for row in selected], ["Q001", "Q003", "Q006"])

    def test_explicit_selection_and_validation(self):
        rows = choose_queries(self.retrieval, self.metrics, ["Q006", "Q001", "Q006"], 3)
        self.assertEqual([row["id"] for row in rows], ["Q006", "Q001"])
        with self.assertRaisesRegex(ValueError, "Unknown query IDs"):
            choose_queries(self.retrieval, self.metrics, ["Q099"], 3)

    def test_preview_filename_matches_judge(self):
        expected = hashlib.sha256(b"example-id").hexdigest()[:20] + ".jpg"
        self.assertEqual(preview_path(Path("run"), "example-id"),
                         Path("run") / "previews" / expected)

    def test_render_saves_preview_with_missing_judgment(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "run"
            run_dir.mkdir()
            preview_file = preview_path(run_dir, "panorama-1")
            preview_file.parent.mkdir()
            Image.new("RGB", (240, 160), "green").save(preview_file)
            retrieval = {"queries": [{
                "id": "Q001", "query": "A green hillside road", "theme": "landscape",
                "matches": [{"image_id": "panorama-1", "rank": 1,
                             "latitude": 1.0, "longitude": 2.0}],
                "random_controls": [{"image_id": "panorama-2",
                                     "latitude": 3.0, "longitude": 4.0}],
            }]}
            (run_dir / "retrieval.json").write_text(json.dumps(retrieval), encoding="utf-8")
            with (run_dir / "per_query.csv").open("w", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["query_id", "P@10",
                                  "mean_relevance@10", "nDCG@10"])
                writer.writeheader()
                writer.writerow({"query_id": "Q001", "P@10": .5,
                                 "mean_relevance@10": 1.5, "nDCG@10": .8})
            (run_dir / "judgments.jsonl").write_text(json.dumps({
                "query_id": "Q001", "image_id": "panorama-1",
                "holistic_score": 2, "category_scores": {"landscape": {"score": 2}},
                "summary": "Hills visible",
            }) + "\n", encoding="utf-8")
            out = Path(folder) / "examples"
            inspect(run_dir, ["Q001"], top_k=1, controls=1, save_dir=out, show=False)
            self.assertTrue((out / "Q001.png").is_file())


if __name__ == "__main__":
    unittest.main()
