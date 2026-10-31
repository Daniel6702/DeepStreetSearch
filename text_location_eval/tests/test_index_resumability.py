"""Checkpoint progress tests without GPU/torch or the Street View dataset."""

import pickle
import random
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from text_location_eval.indexing import _load_parts, _save_part, _selected_keys


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), 6, 4)
        self.ndim = 3

    def reshape(self, *shape):
        return self

    def half(self):
        return self

    def cpu(self):
        return self


class FakeTorch:
    @staticmethod
    def cat(chunks, dim=0):
        assert dim == 0
        return FakeTensor([row for chunk in chunks for row in chunk.rows])

    @staticmethod
    def save(value, path):
        with Path(path).open("wb") as stream:
            pickle.dump(value, stream)

    @staticmethod
    def load(path, **kwargs):
        with Path(path).open("rb") as stream:
            return pickle.load(stream)


class IndexResumabilityTests(unittest.TestCase):
    def test_selected_keys_match_original_sampling(self):
        metadata = {str(i): {} for i in range(20)}
        self.assertEqual(
            _selected_keys(metadata, 7, 42),
            set(random.Random(42).sample(list(metadata), 7)),
        )
        self.assertEqual(_selected_keys(metadata, 20, 42), set(metadata))

    def test_atomic_parts_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            parts = Path(directory)
            signature = {"checkpoint_sha256": "unchanged", "num_images": 3}
            first = parts / "part-000000.pt"
            second = parts / "part-000001.pt"
            _save_part(FakeTorch, first, signature,
                       [{"image_id": "A"}, {"image_id": "B"}],
                       [FakeTensor([1]), FakeTensor([2])])
            self.assertFalse(first.with_suffix(".tmp").exists())
            existing, completed = _load_parts(FakeTorch, parts, signature, {"A", "B", "C"})
            self.assertEqual(existing, [first])
            self.assertEqual(completed, {"A", "B"})
            # The resumed job should encode only C, then write the next part.
            _save_part(FakeTorch, second, signature,
                       [{"image_id": "C"}], [FakeTensor([3])])
            existing, completed = _load_parts(FakeTorch, parts, signature, {"A", "B", "C"})
            self.assertEqual(existing, [first, second])
            self.assertEqual(completed, {"A", "B", "C"})

    def test_wrong_signature_and_out_of_sample_parts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parts = Path(directory)
            _save_part(FakeTorch, parts / "part-000000.pt", {"seed": 42},
                       [{"image_id": "A"}], [FakeTensor([1])])
            with self.assertRaisesRegex(ValueError, "different settings"):
                _load_parts(FakeTorch, parts, {"seed": 43}, {"A"})
            with self.assertRaisesRegex(ValueError, "outside this sample"):
                _load_parts(FakeTorch, parts, {"seed": 42}, {"B"})

    def test_duplicate_and_missing_part_numbers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parts = Path(directory)
            signature = {"seed": 42}
            _save_part(FakeTorch, parts / "part-000000.pt", signature,
                       [{"image_id": "A"}], [FakeTensor([1])])
            _save_part(FakeTorch, parts / "part-000001.pt", signature,
                       [{"image_id": "A"}], [FakeTensor([2])])
            with self.assertRaisesRegex(ValueError, "Duplicate panorama"):
                _load_parts(FakeTorch, parts, signature, {"A"})
            (parts / "part-000001.pt").rename(parts / "part-000002.pt")
            with self.assertRaisesRegex(ValueError, "missing or unexpected"):
                _load_parts(FakeTorch, parts, signature, {"A"})

    def test_full_indexing_resumes_without_reencoding_saved_images(self):
        from argparse import Namespace
        from text_location_eval.indexing import build_index

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset"
            dataset_path.mkdir()
            (dataset_path / "metadata.csv").write_text("id\nA\nB\nC\nD\n")
            grid_path = root / "world.aggrid"
            grid_path.write_text("grid")
            checkpoint_path = root / "checkpoint.pt"
            FakeTorch.save({"model": {}}, checkpoint_path)
            args = Namespace(
                checkpoint=str(checkpoint_path), grid=str(grid_path),
                dataset=str(dataset_path), model="fake", seed=42, num_images=4,
                device="cpu", batch_size=2, workers=0, save_every=2,
                rebuild=False, output=str(root / "reference.pt"),
            )
            metadata = {
                key: {"image_path": f"shard.tar::{key}_360.jpg", "lat": "1",
                      "lon": "2", "__source_shard__": "shard.tar"}
                for key in "ABCD"
            }
            encoded = []

            class FakeDataset:
                interrupt_next = True

                def __init__(self, *args, metadata, **kwargs):
                    self.metadata = metadata
                    self.selected_keys = None  # Matches fraction=1 loader behavior.

                @staticmethod
                def load_metadata(*args):
                    return metadata

                def loader(self, batch_size, **kwargs):
                    # Explicit filtering is vital: without it the real loader
                    # would decode all shard images, including completed ones.
                    assert self.selected_keys == set(self.metadata)
                    keys = sorted(self.selected_keys)
                    for start in range(0, len(keys), batch_size):
                        if start > 0 and FakeDataset.interrupt_next:
                            FakeDataset.interrupt_next = False
                            raise RuntimeError("simulated scheduler interruption")
                        batch_keys = keys[start:start + batch_size]
                        encoded.extend(batch_keys)
                        crops = [[types.SimpleNamespace(image=types.SimpleNamespace(close=lambda: None))
                                  for _ in range(6)] for key in batch_keys]
                        yield crops, [self.metadata[key] for key in batch_keys]

            class FakeProcessor:
                @staticmethod
                def from_pretrained(*args):
                    return FakeProcessor()

                def __call__(self, **kwargs):
                    return {}

            class FakeModel:
                def __init__(self, *args, **kwargs):
                    pass

                def to(self, *args):
                    return self

                def eval(self):
                    return self

                def load_state_dict(self, weights):
                    pass

                def encode_images(self, inputs, batch_size, num_views):
                    return None, FakeTensor(list(range(batch_size)))

            def module(name, **attributes):
                result = types.ModuleType(name)
                result.__dict__.update(attributes)
                return result

            fake_modules = {
                "torch": module(
                    "torch", load=FakeTorch.load, save=FakeTorch.save,
                    cat=FakeTorch.cat, inference_mode=nullcontext,
                    is_tensor=lambda value: False,
                ),
                "tqdm": module("tqdm", tqdm=lambda iterable, **kwargs: iterable),
                "transformers": module("transformers", AutoProcessor=FakeProcessor),
                "adaptive_geogrid": module("adaptive_geogrid", load_grid=lambda path: None),
                "config": module("config", NUM_VIEWS=6),
                "modules.dataset": module("modules.dataset", PanoramaDataset=FakeDataset),
                "modules.dataset.shards": module(
                    "modules.dataset.shards", SOURCE_SHARD="__source_shard__",
                    image_key=lambda filename: Path(filename).stem.removesuffix("_360"),
                ),
                "modules.model": module("modules.model", GeoSearchModel=FakeModel),
            }
            with patch.dict("sys.modules", fake_modules):
                with self.assertRaisesRegex(RuntimeError, "scheduler interruption"):
                    build_index(args)
                self.assertEqual(len(list((root / "reference.parts").glob("*.pt"))), 1)
                self.assertFalse((root / "reference.pt").exists())
                build_index(args)

            result = FakeTorch.load(root / "reference.pt")
            self.assertEqual(len(result["records"]), 4)
            self.assertEqual(result["embeddings"].shape, (4, 6, 4))
            self.assertEqual(sorted(encoded), list("ABCD"))


if __name__ == "__main__":
    unittest.main()
