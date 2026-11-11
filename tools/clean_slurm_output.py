"""Keep scientific results on stdout; send routine progress and library logs to stderr.

Use from a Slurm script via `source experiments/clean_slurm_output.bash`.
Everything excluded from .out is still available in Slurm's .err file.
"""

from __future__ import annotations

import json
import re
import sys


IMPORTANT_STARTS = (
    "Experiment:",
    "Panorama architecture smoke checks passed",
    "Loaded geographic labels:",
    "Grid classes fine -> coarse:",
    "Descriptions:",
    "Description mode:",
    "Revised contrastive positives:",
    "Frozen text encoder:",
    "Geographic branch:",
    "Panorama contrastive loss:",
    "Retrieval fusion:",
    "Dataset:",
    "LR schedule:",
    "Separate learning rates:",
    "Max gradient norm:",
    "Resuming from epoch ",
    "Epoch ",
    "Evaluating ",
    "Median:",
    "Within distance:",
    "Grid hierarchy top-1:",
    "Fine cell top-k:",
    "Saved: ",
    "Saved retrieval for ",
    "Saved finite embeddings to ",
    "Paired retrieval ",
    "Judgments saved in ",
    "Judging existing retrieval:",
    "Saved metrics in ",
    "Qwen evaluation complete:",
    "Qwen evaluation saved:",
    "Existing Qwen metrics:",
    "Evaluation including Qwen ",
    "Weighted semantic retrieval sweep complete:",
    "Full semantic-token evaluation complete:",
    "Corrected cross-view ",
    "Weighted Qwen evaluation complete:",
)


def important(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith(IMPORTANT_STARTS):
        return True
    if re.match(r"Saved [\d,]+ panoramas / [\d,]+ retrieval vectors", stripped):
        return True
    # Keep aggregate failure/coverage warnings, not one warning per image.
    return bool(re.match(r"WARNING: \d+ (unresolved model outputs|incomplete queries)", stripped))


def emit_json(lines: list[str]) -> None:
    block = "".join(lines)
    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        payload = None
    target = sys.stdout if isinstance(payload, dict) and "nDCG@10" in payload else sys.stderr
    target.write(block)
    target.flush()


def main() -> None:
    json_lines: list[str] = []
    for line in sys.stdin:
        if json_lines:
            json_lines.append(line)
            if line.strip() == "}":
                emit_json(json_lines)
                json_lines = []
            continue
        if line.strip() == "{":
            json_lines = [line]
            continue
        target = sys.stdout if important(line) else sys.stderr
        target.write(line)
        target.flush()
    if json_lines:
        sys.stderr.writelines(json_lines)


if __name__ == "__main__":
    main()
