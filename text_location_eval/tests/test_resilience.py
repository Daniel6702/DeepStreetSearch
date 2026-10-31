import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from text_location_eval.common import load_jsonl, write_json
from text_location_eval.judging import judge_retrieval, preview_path
from text_location_eval.metrics import calculate_metrics, summarize


class ResilienceTests(unittest.TestCase):
    def test_invalid_visual_output_is_logged_and_resumable(self):
        retrieval = {
            "index_count": 2,
            "aggregation": "mean",
            "queries": [{
                "id": "Q1", "query": "A street with brick houses",
                "matches": [{"image_id": "A", "rank": 1}, {"image_id": "B", "rank": 2}],
                "random_controls": [],
            }],
        }

        class FakeJudge:
            fail_blue = True

            def __init__(self, args):
                pass

            def generate(self, prompts, images=None):
                if images is None:
                    return [
                        '{"categories":{"architecture":{"importance":3,'
                        '"criteria":["brick houses"]}}}'
                        for _ in prompts
                    ]
                result = []
                for image in images:
                    is_blue = image.getpixel((0, 0))[2] > image.getpixel((0, 0))[0]
                    if is_blue and self.fail_blue:
                        result.append('{"holistic_score":99,"category_scores":{"architecture":99}}')
                    else:
                        result.append('{"holistic_score":2,"category_scores":{"architecture":2}}')
                return result

        def fake_previews(data, directory):
            directory.mkdir(parents=True, exist_ok=True)
            for query in data["queries"]:
                for item in query["matches"]:
                    color = (255, 0, 0) if item["image_id"] == "A" else (0, 0, 255)
                    image = Image.new("RGB", (2, 2), color)
                    try:
                        image.save(preview_path(directory, item["image_id"]), format="PNG")
                    finally:
                        image.close()

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            retrieval_path = folder / "retrieval.json"
            write_json(retrieval_path, retrieval)
            args = SimpleNamespace(retrieval=str(retrieval_path), output=str(folder),
                                   judge_model="fake-judge", judge_batch_size=2)

            with patch("text_location_eval.judging.QwenJudge", FakeJudge), \
                 patch("text_location_eval.judging.prepare_previews", fake_previews):
                judge_retrieval(args)

                self.assertEqual(len(load_jsonl(folder / "judgments.jsonl")), 1)
                failures = load_jsonl(folder / "judge_failures.jsonl")
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0]["image_id"], "B")

                analyses = {item["query_id"]: item
                            for item in load_jsonl(folder / "query_analysis.jsonl")}
                judgments = {(item["query_id"], item["image_id"]): item
                             for item in load_jsonl(folder / "judgments.jsonl")}
                partial = calculate_metrics(retrieval, analyses, judgments)
                self.assertEqual(partial["summary"]["status"], "partial")
                self.assertIsNone(partial["summary"]["nDCG@10"])
                self.assertEqual(partial["summary"]["judgment_coverage"], 0.5)

                # Even with zero complete queries, metrics must write usable files.
                summarize(SimpleNamespace(
                    retrieval=str(retrieval_path),
                    analysis=str(folder / "query_analysis.jsonl"),
                    judgments=str(folder / "judgments.jsonl"),
                    output=str(folder),
                ))
                self.assertEqual(json.loads((folder / "metrics.json").read_text())
                                 ["summary"]["status"], "partial")
                self.assertTrue((folder / "per_query.csv").exists())

                FakeJudge.fail_blue = False
                judge_retrieval(args)

                judgments = {(item["query_id"], item["image_id"]): item
                             for item in load_jsonl(folder / "judgments.jsonl")}
                self.assertEqual(len(judgments), 2)
                complete = calculate_metrics(retrieval, analyses, judgments)
                self.assertEqual(complete["summary"]["status"], "complete")
                self.assertEqual(complete["summary"]["judgment_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
