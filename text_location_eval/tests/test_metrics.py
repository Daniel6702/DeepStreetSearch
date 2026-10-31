import unittest

from text_location_eval.common import parse_json_response
from text_location_eval.judging import parse_with_retry, validate_analysis, validate_judgment
from text_location_eval.metrics import calculate_metrics, ndcg_at_k


class MetricsTests(unittest.TestCase):
    def test_ndcg_perfect_ranking_and_wrong_ranking(self):
        pool = {"great": 3, "okay": 2, "bad": 0}
        self.assertAlmostEqual(ndcg_at_k(["great", "okay"], pool, 2), 1.0)
        self.assertLess(ndcg_at_k(["bad", "okay"], pool, 2), 1.0)
        with self.assertRaises(ValueError):
            ndcg_at_k(["unjudged"], pool, 1)

    def test_ndcg_is_not_absolute_relevance(self):
        # Demonstrates why nDCG must be accompanied by absolute relevance scores.
        self.assertEqual(ndcg_at_k(["weak"], {"weak": 1, "poor": 0}, 1), 1.0)
        self.assertEqual(ndcg_at_k(["all_bad"], {"all_bad": 0}, 1), 0.0)

    def test_query_only_category_mask_and_validation(self):
        categories = validate_analysis({"categories": {
            "architecture": {"importance": 3, "criteria": ["brick houses"]},
            "vegetation": {"importance": 1, "criteria": ["pine trees"]},
        }})
        self.assertNotIn("weather_lighting", categories)
        valid = validate_judgment({
            "holistic_score": 2,
            "category_scores": {
                "architecture": {"score": 3, "evidence": "brick houses visible"},
                "vegetation": {"score": 1, "evidence": "few trees"},
            },
            "summary": "Good buildings, weak vegetation",
        }, categories)
        self.assertEqual(valid["category_scores"]["vegetation"]["score"], 1)
        with self.assertRaises(ValueError):
            validate_judgment({"holistic_score": 2, "category_scores": {
                "architecture": {"score": 3}}}, categories)

    def test_compact_numeric_judgments_are_accepted(self):
        categories = validate_analysis({"categories": {
            "architecture": {"importance": "3", "criteria": "brick facades"},
            "infrastructure": {"importance": 2.0, "criteria": ["wide road"]},
        }})
        result = validate_judgment({
            "holistic_score": "0",
            "category_scores": {"architecture": 0, "infrastructure": {"score": "0"}},
            "summary": "No matching infrastructure",
        }, categories)
        self.assertEqual(result["holistic_score"], 0)
        self.assertEqual(result["category_scores"]["architecture"],
                         {"score": 0, "evidence": ""})
        with self.assertRaises(ValueError):
            validate_judgment({"holistic_score": 1.7, "category_scores": {
                "architecture": 0, "infrastructure": 0}}, categories)
        with self.assertRaises(ValueError):
            validate_judgment({"holistic_score": 2, "category_scores": {
                "architecture": 99, "infrastructure": 0}}, categories)

    def test_parse_fenced_json(self):
        self.assertEqual(parse_json_response('```json\n{"hello":1}\n```'), {"hello": 1})

    def test_parse_missing_final_brace_without_inventing_content(self):
        response = (
            '{"categories":{"architecture":{"importance":3,'
            '"criteria":["timber houses"]},"focus_objects":{"importance":2,'
            '"criteria":["bicycles near entrances"]}}'
        )
        parsed = parse_json_response(response)
        self.assertEqual(parsed["categories"]["focus_objects"]["importance"], 2)
        self.assertEqual(parse_json_response('{"a":[1,2'), {"a": [1, 2]})
        with self.assertRaises(ValueError):
            parse_json_response('{"a":1 "b":2}')
        with self.assertRaises(ValueError):
            parse_json_response('{"a":"unfinished')

    def test_retry_uses_previous_validation_error(self):
        class FakeJudge:
            responses = iter([
                '{"categories":{"not_a_category":{"importance":2,"criteria":["x"]}}}',
                '{"categories":{"architecture":{"importance":2,"criteria":["brick"]}}}',
            ])
            prompts = []

            def generate(self, prompts, images=None):
                self.prompts.append(prompts[0])
                return [next(self.responses)]

        judge = FakeJudge()
        categories = parse_with_retry(judge, "Original query", validate_analysis)
        self.assertIn("architecture", categories)
        self.assertEqual(len(judge.prompts), 2)
        self.assertIn("not_a_category", judge.prompts[1])

    def test_aggregate_skips_inapplicable_categories(self):
        retrieval = {
            "index_count": 20, "aggregation": "mean",
            "queries": [{
                "id": "Q1", "theme": "rural", "geo_context": "none",
                "matches": [
                    {"image_id": "A", "rank": 1},
                    {"image_id": "B", "rank": 2},
                ],
                "random_controls": [{"image_id": "C"}],
            }],
        }
        analyses = {"Q1": {"categories": {
            "landscape": {"importance": 3, "criteria": ["fields"]},
            "infrastructure": {"importance": 1, "criteria": ["road"]},
        }}}
        judgments = {
            ("Q1", "A"): {"holistic_score": 3, "category_scores": {
                "landscape": {"score": 3}, "infrastructure": {"score": 2}}},
            ("Q1", "B"): {"holistic_score": 1, "category_scores": {
                "landscape": {"score": 0}, "infrastructure": {"score": 2}}},
            ("Q1", "C"): {"holistic_score": 0, "category_scores": {}},
        }
        result = calculate_metrics(retrieval, analyses, judgments)
        summary = result["summary"]
        self.assertEqual(summary["P@5"], 0.5)
        self.assertEqual(summary["Success@10"], 1.0)
        self.assertAlmostEqual(summary["mean_weighted_categories@10"], 1.625)
        self.assertEqual({c["category"] for c in result["per_category"]},
                         {"landscape", "infrastructure"})

    def test_all_unjudged_queries_report_partial_results(self):
        retrieval = {
            "index_count": 20, "aggregation": "mean",
            "queries": [{"id": "Q1", "matches": [{"image_id": "A", "rank": 1}],
                         "random_controls": []}],
        }
        result = calculate_metrics(retrieval, {}, {})
        self.assertEqual(result["per_query"], [])
        self.assertEqual(result["summary"]["num_queries_skipped"], 1)
        self.assertEqual(result["summary"]["judgment_coverage"], 0.0)
        self.assertEqual(result["summary"]["status"], "partial")
        self.assertIsNone(result["summary"]["P@10"])


if __name__ == "__main__":
    unittest.main()
