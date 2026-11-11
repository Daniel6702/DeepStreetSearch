"""Small CPU-only checks for the external-test + training-distractor gallery."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools.build_mixed_eval_gallery import make_gallery, read_metadata
from tools.select_external_paired_ids import choose_ids


class MixedGalleryTests(unittest.TestCase):
    def create_dataset(self, root: Path, label: str, count: int):
        images = root / "images"
        images.mkdir(parents=True)
        (images / "shard_000.tar").write_bytes(b"placeholder shard")
        with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_path", "panoid", "lat", "lon"])
            writer.writeheader()
            for i in range(count):
                writer.writerow({"image_path": f"shard_000.tar::{label}{i}_360.jpg",
                                 "panoid": f"{label}{i}", "lat": 1, "lon": 2})

    def test_mixed_gallery_external_positives_and_reruns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_dataset(root / "train", "train", 9)
            self.create_dataset(root / "test", "test", 3)
            gallery = root / "gallery"
            result = make_gallery(root / "train", root / "test", gallery, 8, 42)
            self.assertEqual((result["test_count"], result["train_count"]), (3, 5))
            columns, records = read_metadata(gallery)
            self.assertEqual(len(records), 8)
            self.assertEqual(set(records) & {f"test{i}" for i in range(3)},
                             {f"test{i}" for i in range(3)})
            self.assertEqual(len(set(records) & {f"train{i}" for i in range(9)}), 5)
            self.assertTrue((gallery / "images/shard_test_000.tar").is_symlink())
            self.assertTrue((gallery / "images/shard_train_000.tar").is_symlink())
            self.assertTrue(all(row["image_path"].startswith("shard_") for row in records.values()))
            self.assertEqual(result, make_gallery(root / "train", root / "test", gallery, 8, 42))
            with self.assertRaises(ValueError):
                make_gallery(root / "train", root / "test", gallery, 7, 42)

    def test_overlapping_test_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_dataset(root / "train", "shared", 4)
            self.create_dataset(root / "test", "shared", 2)
            with self.assertRaisesRegex(ValueError, "overlap"):
                make_gallery(root / "train", root / "test", root / "gallery", 5, 42)

    def test_common_paired_ids_are_present_in_both_galleries(self):
        index_20k = [{"image_id": "test1"}, {"image_id": "train1"},
                     {"image_id": "test2"}, {"image_id": "test3"}]
        index_200k = [*index_20k, {"image_id": "test4"}, {"image_id": "train2"}]
        external = {"test1", "test2", "test3", "test4"}
        selected = choose_ids(index_20k, external, 2, 42)
        self.assertEqual(selected, choose_ids(index_20k, external, 2, 42))
        self.assertTrue(set(selected) <= {row["image_id"] for row in index_200k})


if __name__ == "__main__":
    unittest.main()
