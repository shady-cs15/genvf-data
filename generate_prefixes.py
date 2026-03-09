#!/usr/bin/env python3
"""Generate prefixes by truncating reasoning chains from model traces.

Usage:
    python generate_prefixes.py --model qwen3-235b-thinking --start 0 --end 1
    python generate_prefixes.py --model gpt5-mini --start 0 --end 10
    python generate_prefixes.py --model gpt5-mini  # all records
"""

import json
import re
import random
import argparse
from pathlib import Path

from make_prefix import TrajectorySegmenter


# Models where reasoning is in a separate field (thinking models with <think> tags)
SEPARATE_REASONING_MODELS = {"qwen3-235b-thinking", "qwen3.5-27b", "qwen3.5-35b-a3b"}


def get_reasoning_text(record: dict, model: str) -> str | None:
    """Extract the reasoning text from a trace record.

    For thinking models (Qwen3): use the 'reasoning' field.
    For other models (GPT-5-mini): use the 'response' field (CoT is inline).
    Returns None if no usable text is found.
    """
    if model in SEPARATE_REASONING_MODELS:
        text = record.get("reasoning")
        if text:
            return text
        # Fall back to response if reasoning is empty
        return record.get("response")
    else:
        return record.get("response")


def truncate_at_sentence(text: str, rng: random.Random) -> tuple[str, int]:
    """Truncate text at a random sentence boundary.

    Returns (prefix, sentence_index).
    """
    # Split on sentence endings followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= 1:
        # Last resort: truncate at a random character position (50-90%)
        cut = rng.randint(len(text) // 2, max(len(text) - 1, len(text) // 2))
        return text[:cut], 0

    index = rng.randint(1, len(sentences) - 1)
    prefix = " ".join(sentences[:index])
    return prefix, index


def generate_prefix(record: dict, model: str, segmenter: TrajectorySegmenter, rng: random.Random) -> dict:
    """Generate a prefix record from a single trace record."""
    base = {
        "index": record["index"],
        "problem": record["problem"],
        "answer": record.get("answer"),
        "source": record.get("source"),
        "mean_reward": record.get("mean_reward"),
        "full_response": record.get("response"),
        "full_reasoning": record.get("reasoning"),
        "model": record.get("model"),
    }

    text = get_reasoning_text(record, model)

    if not text or not text.strip():
        return {
            **base,
            "prefix": None,
            "prefix_end_index": None,
            "num_thoughts": 0,
            "prefix_type": "skipped",
            "prefix_type_description": "No reasoning or response content available to truncate",
        }

    thoughts = segmenter.segment_trajectory(text)
    thought_contents = [t.content for t in thoughts]
    num_thoughts = len(thought_contents)

    if num_thoughts >= 2:
        index = rng.randint(1, num_thoughts - 1)
        prefix = "\n\n".join(thought_contents[:index])
        return {
            **base,
            "prefix": prefix,
            "prefix_end_index": index,
            "num_thoughts": num_thoughts,
            "prefix_type": "thought_boundary",
            "prefix_type_description": f"Truncated at thought block {index} of {num_thoughts} using transition keyword segmentation",
        }
    else:
        # Single thought block — fall back to sentence boundary
        prefix, sent_idx = truncate_at_sentence(text, rng)
        return {
            **base,
            "prefix": prefix,
            "prefix_end_index": sent_idx,
            "num_thoughts": 1,
            "prefix_type": "sentence_boundary",
            "prefix_type_description": "Single thought block; fell back to truncation at a random sentence boundary",
        }


def main():
    parser = argparse.ArgumentParser(description="Generate prefixes from model traces")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen3-235b-thinking", "gpt5-mini", "qwen3.5-27b", "qwen3.5-35b-a3b"],
        help="Model whose traces to process",
    )
    parser.add_argument("--input-dir", type=str, default="traces")
    parser.add_argument("--output-dir", type=str, default="prefixes")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input_dir) / f"{args.model}.jsonl"
    output_path = Path(args.output_dir) / f"{args.model}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Trace file not found: {input_path}")

    # Load all records, keyed by index
    records = {}
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records[r["index"]] = r

    # Determine range
    all_indices = sorted(records.keys())
    start = args.start
    end = args.end if args.end is not None else max(all_indices) + 1
    indices = [i for i in all_indices if start <= i < end]

    print(f"Model:   {args.model}")
    print(f"Input:   {input_path} ({len(records)} records)")
    print(f"Range:   [{start}, {end})")
    print(f"To process: {len(indices)}")
    print(f"Seed:    {args.seed}")
    print(f"Output:  {output_path}")
    print()

    rng = random.Random(args.seed)
    segmenter = TrajectorySegmenter()

    stats = {"thought_boundary": 0, "sentence_boundary": 0, "skipped": 0}

    with open(output_path, "a") as f:
        for idx in indices:
            record = records[idx]
            result = generate_prefix(record, args.model, segmenter, rng)
            stats[result["prefix_type"]] += 1
            f.write(json.dumps(result) + "\n")

    print(f"Done! Processed {len(indices)} records.")
    print(f"  Thought boundary: {stats['thought_boundary']}")
    print(f"  Sentence boundary: {stats['sentence_boundary']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
