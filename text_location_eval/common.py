"""Small, dependency-free utilities shared by the evaluation stages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


CATEGORIES = (
    "infrastructure", "architecture", "landscape", "vegetation",
    "focus_objects", "weather_lighting", "atmosphere", "other",
)
RUBRIC_VERSION = "text-location-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{lineno}") from exc
    return rows


def load_queries(path: str | Path) -> list[dict]:
    rows = load_jsonl(path)
    ids = set()
    for row in rows:
        query_id, query = row.get("id"), row.get("query")
        if not isinstance(query_id, str) or not query_id or query_id in ids:
            raise ValueError(f"Missing or duplicate query id: {query_id!r}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Empty query for {query_id}")
        ids.add(query_id)
    if not rows:
        raise ValueError(f"No queries in {path}")
    return rows


def write_json(path: str | Path, data: object) -> None:
    """Write complete results atomically so interrupted runs don't corrupt them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")
        stream.flush()


def _complete_json_suffix(text: str) -> str:
    """Restore only missing final ]/} delimiters, never missing values or quotes."""
    stack = []
    pairs = {"{": "}", "[": "]"}
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif char in ("]", "}"):
            if not stack or stack.pop() != char:
                raise ValueError("Judge JSON contains mismatched closing delimiters")
    if in_string:
        raise ValueError("Judge JSON ends inside a quoted string")
    return text + "".join(reversed(stack))


def parse_json_response(text: str) -> dict:
    """Parse Qwen JSON, optionally closing unfinished outer objects or arrays."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"Judge did not return a JSON object: {text[:160]!r}")
    candidate = text[start:]
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(candidate)
    except json.JSONDecodeError:
        completed = _complete_json_suffix(candidate)
        if completed == candidate:
            raise
        payload, _ = decoder.raw_decode(completed)
    if not isinstance(payload, dict):
        raise ValueError("Judge JSON must be an object")
    return payload
