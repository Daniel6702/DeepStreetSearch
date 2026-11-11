"""Build a repeatable training/evaluation registry from existing run artifacts.

Run from the repository root:
    python tools/collect_experiments.py

Only small JSON files, Slurm logs and checkpoint metadata are read. Gallery
indexes, images and complete checkpoint tensors are never materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = re.compile(r"^(?P<stem>.+)-(?P<hash>[0-9a-f]{12})$")
EPOCH_LINE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?:\d+)\s+-\s+loss:\s*(?P<loss>[-+\deE.]+)"
    r"\s+-\s+geo:\s*(?P<geographic>[-+\deE.]+)"
    r"\s+-\s+contrastive:\s*(?P<contrastive>[-+\deE.]+)"
    r"(?:\s+-\s+panorama:\s*(?P<panorama>[-+\deE.]+))?"
)
JOB_NAME = re.compile(r"^#SBATCH\s+--job-name=(\S+)", re.MULTILINE)
CHECKPOINT_VAR = re.compile(r"^CHECKPOINT=[\"']?([^\"'\n]+)", re.MULTILINE)


def path_label(path: Path, root: Path) -> str:
    """Prefer paths relative to the repository to keep reports relocatable."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_json(path: Path, warnings: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"Cannot read {path}: {exc}")
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_json(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): as_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [as_json(item) for item in value]
    return str(value)


def checkpoint_metadata(path: Path) -> dict:
    """Read torch metadata with memory-mapped tensor storage (no huge RAM copy)."""
    try:
        import torch
    except ImportError:
        return {"status": "unavailable", "reason": "PyTorch is not installed"}
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(checkpoint, dict):
            return {"status": "unavailable", "reason": "Not a training checkpoint dictionary"}
        metadata = {
            "status": "loaded",
            "completed_epochs": checkpoint.get("epoch", -1) + 1
            if isinstance(checkpoint.get("epoch"), int) else None,
            "global_step": checkpoint.get("global_step"),
            "parameters": as_json(checkpoint.get("run_config") or {}),
        }
        if not checkpoint.get("run_config"):
            metadata["note"] = "This checkpoint does not contain run_config."
        del checkpoint
        return metadata
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def existing_hash(path: Path, previous: dict) -> str:
    """Reuse digest only when size and nanosecond modification time agree."""
    stat = path.stat()
    if (previous.get("size_bytes") == stat.st_size
            and previous.get("mtime_ns") == stat.st_mtime_ns
            and re.fullmatch(r"[0-9a-f]{64}", previous.get("sha256", ""))):
        return previous["sha256"]
    return sha256(path)


def read_epoch_log(path: Path) -> list[dict]:
    """Only complete epoch summary lines are recorded, not progress-bar noise."""
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    epochs = {}
    for match in EPOCH_LINE.finditer(data):
        row = {"epoch": int(match.group("epoch"))}
        for key in ("loss", "geographic", "contrastive", "panorama"):
            if match.group(key) is not None:
                row[key] = float(match.group(key))
        epochs[row["epoch"]] = row
    return [epochs[epoch] for epoch in sorted(epochs)]


def discover_training_scripts(root: Path) -> dict[str, list[dict]]:
    """Keep the original Slurm command to fill gaps in older run_config data."""
    scripts = {}
    for script in sorted([*root.glob("*.bash"), *root.glob("experiments/*.bash")]):
        try:
            contents = script.read_text(encoding="utf-8")
        except OSError:
            continue
        checkpoint = CHECKPOINT_VAR.search(contents)
        if not checkpoint or "$" in checkpoint.group(1):
            continue
        lines = contents.splitlines()
        command = None
        for index, line in enumerate(lines):
            if not re.match(r"^\s*python(?:\d)?(?:\s+-u)?\s+train\.py\b", line):
                continue
            parts = [line.strip()]
            following = index + 1
            while parts[-1].endswith("\\") and following < len(lines):
                parts[-1] = parts[-1][:-1].rstrip()
                parts.append(lines[following].strip())
                following += 1
            command = " ".join(parts)
            break
        if command is None:
            continue  # Evaluation/judging-only Slurm jobs are not training sources.
        job_name = JOB_NAME.search(contents)
        variables = {}
        for name in ("TRAIN_DATASET", "TEST_DATASET", "GRID", "LABELS"):
            match = re.search(rf"(?m)^{name}=[\"']([^\"'\n]+)[\"']", contents)
            if match:
                variables[name.lower()] = match.group(1)
        scripts.setdefault(Path(checkpoint.group(1)).name, []).append({
            "path": path_label(script, root),
            "job_name": job_name.group(1) if job_name else None,
            "declared_train_command": command,
            "declared_paths": variables,
            "note": "Slurm script contents describe intended settings; checkpoint run_config is authoritative.",
        })
    return scripts


def discover_training_logs(root: Path) -> dict[str, list[dict]]:
    """Match R-jobname.jobid.out to the Slurm script's checkpoint assignment."""
    job_to_checkpoint = {}
    for script in sorted([*root.glob("*.bash"), *root.glob("experiments/*.bash")]):
        try:
            contents = script.read_text(encoding="utf-8")
        except OSError:
            continue
        job_name, checkpoint = JOB_NAME.search(contents), CHECKPOINT_VAR.search(contents)
        if job_name and checkpoint and "$" not in checkpoint.group(1):
            # Unique job names only; ambiguous mappings are not guessed.
            job_to_checkpoint.setdefault(job_name.group(1), set()).add(
                Path(checkpoint.group(1)).name
            )

    associated = {}
    for path in sorted(root.glob("R-*.out")):
        job_match = re.fullmatch(r"R-(.+)\.(\d+)\.out", path.name)
        if not job_match:
            continue
        candidates = job_to_checkpoint.get(job_match.group(1), set())
        if len(candidates) != 1:
            continue
        history = read_epoch_log(path)
        if history:
            associated.setdefault(next(iter(candidates)), []).append({
                "log": path_label(path, root), "job_id": job_match.group(2),
                "epochs": history,
                "source": "Matched Slurm job name to a script CHECKPOINT assignment",
            })
    return associated


def run_directory(path: Path, evaluation_root: Path) -> Path:
    """Resolve both experiments/<checkpoint>-<hash>/test and older layouts."""
    relative = path.relative_to(evaluation_root)
    if "experiments" in relative.parts:
        index = relative.parts.index("experiments")
        if len(relative.parts) > index + 1:
            return evaluation_root.joinpath(*relative.parts[:index + 2])
    if path.name == "retrieval.json":
        return path.parent.parent if path.parent.name in {"max", "mean"} else path.parent
    return path.parent.parent  # paired-*/summary.json or image-location/summary.json


def discover_evaluation_directories(evaluation_roots: list[Path]) -> list[Path]:
    result = set()
    for evaluation_root in evaluation_roots:
        if not evaluation_root.exists():
            continue
        for path in evaluation_root.rglob("retrieval.json"):
            result.add(run_directory(path, evaluation_root))
        for pattern in ("image-location/summary.json", "paired-*/summary.json"):
            for path in evaluation_root.rglob(pattern):
                result.add(run_directory(path, evaluation_root))
    return sorted(result)


def find_judgments(retrieval_path: Path, root: Path, warnings: list[str]) -> list[dict]:
    retrieval_digest = sha256(retrieval_path)
    candidates = [retrieval_path.parent / f"judge-{retrieval_digest[:12]}",
                  retrieval_path.parent]
    results = []
    for judge_dir in candidates:
        manifest_path = judge_dir / "judge_manifest.json"
        metrics_path = judge_dir / "metrics.json"
        if not manifest_path.is_file() and not metrics_path.is_file():
            continue
        manifest = read_json(manifest_path, warnings) if manifest_path.is_file() else None
        if manifest and manifest.get("retrieval_sha256") != retrieval_digest:
            warnings.append(f"Skipping mismatched Qwen manifest: {manifest_path}")
            continue
        metrics = read_json(metrics_path, warnings) if metrics_path.is_file() else None
        summary = metrics.get("summary", {}) if isinstance(metrics, dict) else None
        results.append({
            "directory": path_label(judge_dir, root),
            "manifest": manifest,
            "metrics_path": path_label(metrics_path, root) if metrics_path.is_file() else None,
            "status": summary.get("status", "unknown") if summary else "judging_without_metrics",
            "summary": summary,
            "per_category": metrics.get("per_category", []) if isinstance(metrics, dict) else [],
            "skipped_queries": metrics.get("skipped_queries", []) if isinstance(metrics, dict) else [],
            "per_query_path": (path_label(judge_dir / "per_query.csv", root)
                               if (judge_dir / "per_query.csv").exists() else None),
            "per_category_path": (path_label(judge_dir / "per_category.csv", root)
                                  if (judge_dir / "per_category.csv").exists() else None),
            "judgments_path": (path_label(judge_dir / "judgments.jsonl", root)
                               if (judge_dir / "judgments.jsonl").exists() else None),
            "query_analysis_path": (path_label(judge_dir / "query_analysis.jsonl", root)
                                    if (judge_dir / "query_analysis.jsonl").exists() else None),
            "failures_path": (path_label(judge_dir / "judge_failures.jsonl", root)
                              if (judge_dir / "judge_failures.jsonl").exists() else None),
        })
    return results


def evaluation_record(run_dir: Path, root: Path, warnings: list[str]) -> dict:
    record = {"directory": path_label(run_dir, root),
              "image_location": [], "paired_retrieval": [], "text_retrieval": []}
    for summary_path in sorted(run_dir.rglob("image-location/summary.json")):
        data = read_json(summary_path, warnings)
        if isinstance(data, dict):
            record["image_location"].append({
                "summary_path": path_label(summary_path, root), "results": data,
                "predictions_path": (path_label(summary_path.with_name("predictions.csv"), root)
                                     if summary_path.with_name("predictions.csv").exists() else None),
            })
    for summary_path in sorted(run_dir.rglob("paired-*/summary.json")):
        data = read_json(summary_path, warnings)
        if isinstance(data, dict):
            record["paired_retrieval"].append({
                "summary_path": path_label(summary_path, root), "results": data,
                "queries_path": (path_label(summary_path.with_name("queries.csv"), root)
                                 if summary_path.with_name("queries.csv").exists() else None),
            })
    for retrieval_path in sorted(run_dir.rglob("retrieval.json")):
        data = read_json(retrieval_path, warnings)
        if not isinstance(data, dict):
            continue
        # Save configuration and outputs without duplicating hundreds of image matches.
        parameters = {key: value for key, value in data.items() if key != "queries"}
        queries = data.get("queries", [])
        dataset_name = Path((parameters.get("index_signature") or {}).get("dataset", "")).name
        relative_parts = retrieval_path.relative_to(run_dir).parts
        split = "external_test" if "test" in relative_parts or dataset_name == "world_test" else "legacy_or_unspecified"
        record["text_retrieval"].append({
            "retrieval_path": path_label(retrieval_path, root),
            "split_hint": split,
            "parameters": parameters,
            "query_count": len(queries),
            "judgments": find_judgments(retrieval_path, root, warnings),
        })
    return record


def evaluation_identity(run_dir: Path, evaluation: dict) -> tuple[str | None, str | None]:
    match = RUN_NAME.fullmatch(run_dir.name)
    if match:
        return match.group("stem"), match.group("hash")
    for geo in evaluation["image_location"]:
        result = geo["results"]
        if result.get("checkpoint_sha256_12"):
            return Path(result.get("checkpoint", "unknown")).stem, result["checkpoint_sha256_12"]
    for retrieval in evaluation["text_retrieval"]:
        signature = retrieval["parameters"].get("index_signature") or {}
        digest = signature.get("checkpoint_sha256")
        if digest:
            return None, digest[:12]
    return None, None


def find_indexes(checkpoint_stem: str, digest: str, root: Path) -> list[dict]:
    directory = root / "outputs/indexes"
    if not directory.exists():
        return []
    prefix = f"{checkpoint_stem}-{digest[:12]}"
    indexes = []
    for path in sorted(directory.glob(f"{prefix}*.pt")):
        if path.stem != prefix and not path.stem.startswith(prefix + "-"):
            continue
        indexes.append({"path": path_label(path, root), "size_bytes": path.stat().st_size})
    return indexes


def load_notes(path: Path, warnings: list[str]) -> dict:
    if not path.exists():
        return {}
    notes = read_json(path, warnings)
    if not isinstance(notes, dict):
        warnings.append(f"Notes file must be a JSON object keyed by checkpoint filename: {path}")
        return {}
    return notes


def collect(root: Path, checkpoint_directories: Path | list[Path], evaluation_roots: list[Path],
            previous: dict | None = None, notes: dict | None = None) -> dict:
    warnings = []
    previous = previous or {}
    notes = notes or {}
    if isinstance(checkpoint_directories, Path):
        checkpoint_directories = [checkpoint_directories]
    previous_checkpoints = {item.get("path"): item for item in previous.get("checkpoints", [])}
    training_logs = discover_training_logs(root)
    training_scripts = discover_training_scripts(root)
    checkpoints = []
    paths = {path.resolve() for directory in checkpoint_directories if directory.exists()
             for path in directory.rglob("*.pt")}
    for path in sorted(paths):
        if path.name.endswith(".tmp") or not path.is_file():
            continue
        relative = path_label(path, root)
        old = previous_checkpoints.get(relative, {})
        try:
            digest = existing_hash(path, old)
        except OSError as exc:
            warnings.append(f"Cannot hash {path}: {exc}")
            continue
        stat = path.stat()
        if old.get("sha256") == digest and old.get("size_bytes") == stat.st_size \
                and old.get("mtime_ns") == stat.st_mtime_ns and "training" in old:
            training = old["training"]
        else:
            training = checkpoint_metadata(path)
        checkpoints.append({
            "name": path.stem, "path": relative, "sha256": digest,
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "training": training, "training_logs": training_logs.get(path.name, []),
            "training_scripts": training_scripts.get(path.name, []),
            "notes": notes.get(path.name, notes.get(path.stem, "")),
            "indexes": find_indexes(path.stem, digest, root),
            "evaluations": [],
        })

    by_hash = {}
    for item in checkpoints:
        by_hash.setdefault(item["sha256"][:12], []).append(item)
    orphan_evaluations = []
    for run_dir in discover_evaluation_directories(evaluation_roots):
        evaluation = evaluation_record(run_dir, root, warnings)
        stem, digest = evaluation_identity(run_dir, evaluation)
        possible = by_hash.get(digest, [])
        matched = [item for item in possible if stem is None or item["name"] == stem]
        # A digest uniquely identifies checkpoint bytes, even if renamed.
        if not matched and len(possible) == 1:
            matched = possible
        if len(matched) == 1:
            matched[0]["evaluations"].append(evaluation)
        else:
            orphan_evaluations.append({"checkpoint_name": stem,
                                       "checkpoint_sha256_12": digest, **evaluation})

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_directories": [path_label(p, root) for p in checkpoint_directories],
        "evaluation_roots": [path_label(p, root) for p in evaluation_roots],
        "checkpoints": checkpoints,
        "orphan_evaluations": orphan_evaluations,
        "warnings": warnings,
        "limitations": [
            "Training parameters come from checkpoint run_config; older checkpoints may omit fields.",
            "Epoch losses come from Slurm stdout only when a unique job-name-to-script match exists.",
            "Raw per-query judgments, predictions and image embeddings remain in the linked source files.",
            "Judge manifest records model and rubric, but not all generation parameters.",
            "Partial Qwen metrics exclude queries with unresolved judgments.",
        ],
    }


def markdown_report(report: dict) -> str:
    def short(value, digits=3):
        return f"{value:.{digits}f}" if isinstance(value, (float, int)) else "—"

    def quote(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# DeepStreetSearch experiment registry", "",
             f"Generated: {report['generated_at_utc']}", "",
             "The JSON registry contains all recorded training parameters and detailed evaluations. ",
             "A dash (—) means the metric has not been recorded.", "",
             "| Checkpoint | Epochs | Test median km ↓ | Paired mean R@1 ↑ | Paired max R@1 ↑ | Text split | Qwen mean nDCG@10 ↑ | Qwen max nDCG@10 ↑ | Qwen mean P@10 ↑ | Qwen coverage |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for checkpoint in report["checkpoints"]:
        geo_summary = None
        paired = {}
        text_metrics = {}
        for evaluation in checkpoint["evaluations"]:
            for item in evaluation["image_location"]:
                if item["results"].get("split") == "external":
                    geo_summary = item["results"]
            for item in evaluation["paired_retrieval"]:
                if item["results"].get("split") == "external":
                    paired[item["results"].get("aggregation")] = item["results"].get("overall", {})
            for item in evaluation["text_retrieval"]:
                aggregation = item["parameters"].get("aggregation")
                if aggregation not in {"mean", "max"}:
                    continue
                for judgment in item["judgments"]:
                    if not judgment["summary"]:
                        continue
                    candidate = (item["split_hint"] == "external_test",
                                 judgment["summary"].get("status") == "complete",
                                 judgment["summary"].get("judgment_coverage") or 0)
                    previous = text_metrics.get(aggregation)
                    if previous is None or candidate > previous[0]:
                        text_metrics[aggregation] = (candidate, judgment["summary"])
        mean_summary = text_metrics.get("mean", (None, {}))[1]
        max_summary = text_metrics.get("max", (None, {}))[1]
        text_split = ("test" if text_metrics.get("mean", ((False,),))[0][0] else "legacy") \
            if mean_summary else "—"
        training = checkpoint.get("training") or {}
        epochs = training.get("completed_epochs")
        row = [quote(checkpoint["name"]), str(epochs) if epochs is not None else "—",
               short((geo_summary or {}).get("median_error_km"), 1),
               short(paired.get("mean", {}).get("recall@1")),
               short(paired.get("max", {}).get("recall@1")),
               text_split, short(mean_summary.get("nDCG@10")),
               short(max_summary.get("nDCG@10")),
               short(mean_summary.get("P@10")),
               short(mean_summary.get("judgment_coverage"))]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Source paths", ""])
    for checkpoint in report["checkpoints"]:
        lines.append(f"### {checkpoint['name']}")
        lines.append("")
        lines.append(f"- Checkpoint: `{checkpoint['path']}`")
        if checkpoint["notes"]:
            lines.append(f"- Notes: {quote(checkpoint['notes'])}")
        for training_script in checkpoint.get("training_scripts", []):
            lines.append(f"- Training script: `{training_script['path']}`")
        for training_log in checkpoint["training_logs"]:
            lines.append(f"- Training log: `{training_log['log']}`")
        for evaluation in checkpoint["evaluations"]:
            lines.append(f"- Evaluation: `{evaluation['directory']}`")
        lines.append("")
    if report["orphan_evaluations"]:
        lines.extend(["## Evaluations without a matching checkpoint", ""])
        for item in report["orphan_evaluations"]:
            lines.append(f"- `{item['directory']}` (checkpoint hash: {item['checkpoint_sha256_12'] or 'unknown'})")
        lines.append("")
    lines.extend(["## Interpretation notes", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    if report["warnings"]:
        lines.extend(["", "## Collection warnings", ""])
        lines.extend(f"- {quote(item)}" for item in report["warnings"])
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoints", type=Path, action="append", default=None,
                        help="Repeat to scan additional checkpoint directories")
    parser.add_argument("--eval-root", type=Path, action="append", default=None,
                        help="Repeat to scan multiple evaluation trees")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/registry.json"))
    parser.add_argument("--notes", type=Path, default=Path("outputs/experiments/notes.json"))
    args = parser.parse_args()
    root = args.root.resolve()

    def within_root(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    checkpoint_directories = [within_root(item) for item in (
        args.checkpoints or [Path("outputs/checkpoints"), Path("checkpoints")]
    )]
    evaluation_roots = [within_root(item) for item in (
        args.eval_root or [Path("outputs/eval"), Path("results/text_location_eval")]
    )]
    output = within_root(args.output)
    previous = read_json(output, []) if output.is_file() else None
    notes = load_notes(within_root(args.notes), [])
    report = collect(root, checkpoint_directories, evaluation_roots, previous, notes)
    atomic_write(output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    md_path = output.with_suffix(".md")
    atomic_write(md_path, markdown_report(report))
    print(f"Found {len(report['checkpoints'])} checkpoints, "
          f"{sum(len(item['evaluations']) for item in report['checkpoints'])} linked evaluation runs, "
          f"{len(report['orphan_evaluations'])} unlinked evaluations")
    print(f"Saved: {path_label(output, root)} and {path_label(md_path, root)}")
    if report["warnings"]:
        print(f"WARNING: {len(report['warnings'])} files could not be read; see registry.json")


if __name__ == "__main__":
    main()
