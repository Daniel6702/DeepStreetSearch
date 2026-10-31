"""Blinded, resumable, category-aware Qwen3-VL panorama relevance judging."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from .common import (
    CATEGORIES, RUBRIC_VERSION, append_jsonl, file_sha256,
    load_jsonl, parse_json_response, write_json,
)


def _integer_grade(value, minimum: int, maximum: int) -> int:
    """Accept unambiguous numeric JSON variants, never invent or clamp a score."""
    if type(value) is int:
        grade = value
    elif type(value) is float and value.is_integer():
        grade = int(value)
    elif isinstance(value, str) and value.strip() in {
        str(grade) for grade in range(minimum, maximum + 1)
    }:
        grade = int(value.strip())
    else:
        raise ValueError(f"Invalid grade: {value!r}")
    if not minimum <= grade <= maximum:
        raise ValueError(f"Grade {grade} outside [{minimum}, {maximum}]")
    return grade


def validate_analysis(payload: dict) -> dict:
    categories = payload.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Query analysis must contain applicable categories")
    cleaned = {}
    for name, item in categories.items():
        if name not in CATEGORIES or not isinstance(item, dict):
            raise ValueError(f"Unknown or malformed category: {name!r}")
        importance = _integer_grade(item.get("importance"), 1, 3)
        criteria = item.get("criteria")
        if isinstance(criteria, str) and criteria.strip():
            criteria = [criteria]
        if not isinstance(criteria, list) or not criteria or not all(
            isinstance(c, str) and c.strip() for c in criteria
        ):
            raise ValueError(f"Missing visual criteria for {name}")
        cleaned[name] = {"importance": importance, "criteria": criteria}
    return cleaned


def validate_judgment(payload: dict, active_categories: dict) -> dict:
    overall = _integer_grade(payload.get("holistic_score"), 0, 3)
    category_scores = payload.get("category_scores")
    if not isinstance(category_scores, dict) or set(category_scores) != set(active_categories):
        raise ValueError("Judge must score every applicable category, and no other categories")
    cleaned = {}
    for name, item in category_scores.items():
        # Qwen often emits {"architecture": 0} instead of
        # {"architecture": {"score": 0, "evidence": "..."}}.
        value = item.get("score") if isinstance(item, dict) else item
        evidence = item.get("evidence", "") if isinstance(item, dict) else ""
        cleaned[name] = {
            "score": _integer_grade(value, 0, 3),
            "evidence": str(evidence),
        }
    return {
        "holistic_score": overall,
        "category_scores": cleaned,
        "summary": str(payload.get("summary", "")),
    }


def query_analysis_prompt(query: str, geo_context: str) -> str:
    return (
        "You are defining a fixed evaluation rubric for a text-to-location search query. "
        "Read ONLY the query, without seeing any retrieved images. Choose only categories "
        "whose visually observable features are mentioned or clearly implied. "
        "Do not penalize categories that the user never requested. "
        "For each applicable category give importance 1 (minor), 2 (important), or "
        "3 (central), and a short list of concrete observable criteria. Do not treat a "
        "place name itself as a visually verifiable property; geographic names are assessed "
        "separately. If the query asks for a mood, put it under atmosphere as subjective. "
        f"Allowed categories: {', '.join(CATEGORIES)}. "
        'Return ONLY JSON: {"categories":{"architecture":{"importance":2,'
        '"criteria":["brick facades"]}}}. Omit all inapplicable categories.\n'
        f"Query: {query}\nGeographic context tag: {geo_context}"
    )


def image_judgment_prompt(query: str, categories: dict) -> str:
    requested = json.dumps(categories, ensure_ascii=False)
    return (
        "Evaluate how well this six-view Street View panorama matches the text query. "
        "All six tiles belong to the SAME location, and a relevant feature visible in any "
        "tile may count. The images may show objects unrelated to the query; ignore those. "
        "Use ONLY what can be seen, not guessed country, city, or GPS coordinates. "
        "Do not infer that the scene is correct merely because its style seems geographically "
        "plausible. Judge the requested features together, including contradictions. "
        "The category list was chosen BEFORE seeing the image; do not add, remove, "
        "or mark categories N/A. Score each on 0=no match/contradiction, 1=weak, "
        "2=good, 3=strong. Separately give an independent holistic 0-3 scene match score. "
        "A missing requested feature warrants a low score; an unrequested feature does not. "
        "A place name is not visual evidence and must not influence these scores. "
        'Return ONLY JSON: {"holistic_score":2,"category_scores":'
        '{"architecture":{"score":2,"evidence":"brick facades visible"}},'
        '"summary":"brief justification"}. Include exactly the category keys provided.\n'
        f"Query: {query}\nApplicable categories and criteria: {requested}"
    )


class QwenJudge:
    def __init__(self, args):
        # Avoid FlashInfer's JIT-compiled sampler: cluster CUDA runtime headers
        # may be absent even when vLLM itself and the GPU work correctly.
        # vLLM then uses its native PyTorch/Triton sampling implementation.
        # Must be set before importing vllm (also when running `judge` alone).
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.processor = AutoProcessor.from_pretrained(args.judge_model, trust_remote_code=True)
        self.llm = LLM(
            model=args.judge_model, tokenizer=args.judge_model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_cache_gb=0,
        )
        self.sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    def generate(self, prompts: list[str], images: list | None = None) -> list[str]:
        if images is not None and len(images) != len(prompts):
            raise ValueError("Number of images and prompts must agree")
        requests = []
        for idx, prompt in enumerate(prompts):
            content = [{"type": "text", "text": prompt}]
            if images is not None:
                content.insert(0, {"type": "image"})
            conversation = [{"role": "user", "content": content}]
            chat = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True)
            request = {"prompt": chat}
            if images is not None:
                request["multi_modal_data"] = {"image": images[idx]}
            requests.append(request)
        output = self.llm.generate(requests, self.sampling)
        return [item.outputs[0].text.strip() for item in output]


def parse_with_retry(judge: QwenJudge, prompt: str, validate, image=None) -> dict:
    previous_error = ""
    for attempt in range(3):
        reminder = (
            "\nYour previous response failed JSON/schema validation: "
            f"{previous_error}. Return one COMPLETE JSON object with all required "
            "keys, balanced braces, and no Markdown or explanation."
            if attempt else ""
        )
        text = judge.generate([prompt + reminder],
                              None if image is None else [image])[0]
        try:
            return validate(parse_json_response(text))
        except (ValueError, json.JSONDecodeError) as exc:
            previous_error = str(exc)
            if attempt == 2:
                raise RuntimeError(
                    f"Judge returned invalid JSON after three attempts: {text[:300]!r}"
                ) from exc
    raise AssertionError("unreachable")


def preview_path(directory: Path, image_id: str) -> Path:
    key = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:20]
    return directory / f"{key}.jpg"


def prepare_previews(retrieval: dict, directory: Path) -> None:
    """Decode only retrieved panoramas, streaming each image shard once."""
    from PIL import Image

    from modules.dataset.shards import image_key, iter_shard_images
    from modules.panorama import split_panorama

    directory.mkdir(parents=True, exist_ok=True)
    by_shard = defaultdict(set)
    for query in retrieval["queries"]:
        for record in [*query["matches"], *query.get("random_controls", [])]:
            if not preview_path(directory, record["image_id"]).exists():
                by_shard[record["shard"]].add(record["image_id"])

    for shard, requested in by_shard.items():
        found = set()
        for filename, image in iter_shard_images(shard, allowed_keys=requested):
            key = image_key(filename)
            crops = []
            try:
                crops = split_panorama(image)
                tile_size = 384
                sheet = Image.new("RGB", (tile_size * 3, tile_size * 2))
                try:
                    for i, crop in enumerate(crops):
                        tile = crop.image.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
                        try:
                            sheet.paste(tile, ((i % 3) * tile_size, (i // 3) * tile_size))
                        finally:
                            tile.close()
                    sheet.save(preview_path(directory, key), "JPEG", quality=85)
                finally:
                    sheet.close()
                found.add(key)
            finally:
                image.close()
                for crop in crops:
                    crop.image.close()
        missing = requested - found
        if missing:
            raise RuntimeError(f"Could not find {len(missing)} indexed panorama(s) in {shard}: "
                               f"{sorted(missing)[:3]}")


def judge_retrieval(args) -> None:
    from PIL import Image

    with Path(args.retrieval).open("r", encoding="utf-8") as stream:
        retrieval = json.load(stream)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "judge_manifest.json"
    manifest = {
        "retrieval_sha256": file_sha256(args.retrieval),
        "judge_model": args.judge_model,
        "rubric": RUBRIC_VERSION,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing judgments use a different retrieval/model/rubric. Choose a new --output.")
    else:
        if (output / "query_analysis.jsonl").exists() or (output / "judgments.jsonl").exists():
            raise ValueError("Existing judgment files have no manifest; choose a new --output")
        write_json(manifest_path, manifest)

    analysis_file = output / "query_analysis.jsonl"
    judgment_file = output / "judgments.jsonl"
    failure_file = output / "judge_failures.jsonl"
    analyses = {row["query_id"]: row for row in load_jsonl(analysis_file)} if analysis_file.exists() else {}
    judgments = {(row["query_id"], row["image_id"]) for row in load_jsonl(judgment_file)} \
        if judgment_file.exists() else set()
    failures_this_run = 0

    preview_dir = output / "previews"
    prepare_previews(retrieval, preview_dir)

    judge = QwenJudge(args)
    for query in retrieval["queries"]:
        qid = query["id"]
        if qid in analyses:
            continue
        prompt = query_analysis_prompt(query["query"], query.get("geo_context", "none"))
        try:
            categories = parse_with_retry(judge, prompt, validate_analysis)
        except RuntimeError as exc:
            failures_this_run += 1
            append_jsonl(failure_file, {
                "stage": "query_analysis", "query_id": qid, "error": str(exc),
            })
            print(f"WARNING: Could not analyze query {qid}; continuing")
            continue
        row = {"query_id": qid, "categories": categories}
        append_jsonl(analysis_file, row)
        analyses[qid] = row
    print(f"Query rubrics ready: {len(analyses)}")

    pending = []
    for query in retrieval["queries"]:
        if query["id"] not in analyses:
            continue
        for record in [*query["matches"], *query.get("random_controls", [])]:
            key = (query["id"], record["image_id"])
            if key not in judgments:
                pending.append((query, record))

    for start in range(0, len(pending), args.judge_batch_size):
        batch = pending[start:start + args.judge_batch_size]
        images = []
        try:
            for _, record in batch:
                with Image.open(preview_path(preview_dir, record["image_id"])) as img:
                    images.append(img.convert("RGB"))
            prompts = [image_judgment_prompt(query["query"],
                       analyses[query["id"]]["categories"]) for query, _ in batch]
            responses = judge.generate(prompts, images)
            for (query, record), prompt, image, response in zip(batch, prompts, images, responses):
                categories = analyses[query["id"]]["categories"]
                try:
                    result = validate_judgment(parse_json_response(response), categories)
                except (ValueError, json.JSONDecodeError):
                    try:
                        result = parse_with_retry(
                            judge, prompt, lambda p: validate_judgment(p, categories), image
                        )
                    except RuntimeError as exc:
                        failures_this_run += 1
                        append_jsonl(failure_file, {
                            "stage": "visual_judgment", "query_id": query["id"],
                            "image_id": record["image_id"], "error": str(exc),
                        })
                        print(f"WARNING: Unresolved judgment for {query['id']} / "
                              f"{record['image_id']}; continuing")
                        continue
                row = {"query_id": query["id"], "image_id": record["image_id"], **result}
                append_jsonl(judgment_file, row)
                judgments.add((query["id"], record["image_id"]))
        finally:
            for image in images:
                image.close()
        print(f"Judged {len(judgments)} query/panorama pairs")
    print(f"Judgments saved in {output}")
    if failures_this_run:
        print(f"WARNING: {failures_this_run} unresolved model outputs recorded in "
              f"{failure_file}. They will be retried on the next run.")
