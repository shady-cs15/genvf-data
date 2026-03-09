#!/usr/bin/env python3
"""Generate traces from models via OpenRouter / OpenAI for GVF-3k dataset.

Usage:
    python generate_traces.py --model qwen3-235b-thinking
    python generate_traces.py --model gpt5-mini --concurrency 8 --temperature 0.7
    python generate_traces.py --model qwen3-235b-thinking --start 0 --end 100  # subset
"""

import asyncio
import json
import os
import argparse
import time
from pathlib import Path

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# ── Model registry ──────────────────────────────────────────────────────────
# Each entry: model_id, base_url, api_key_env
MODELS = {
    "qwen3-235b-thinking": {
        "model_id": "qwen/qwen3-235b-a22b-thinking-2507",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "gpt5-mini": {
        "model_id": "gpt-5-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": None,  # gpt-5-mini only supports default (1)
    },
    "qwen3.5-27b": {
        "model_id": "qwen/qwen3.5-27b",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "qwen3.5-35b-a3b": {
        "model_id": "qwen/qwen3.5-35b-a3b",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}

# ── Token budgets by problem type ────────────────────────────────────────────
MAX_TOKENS_MATH = 32768      # problems with concrete answers
MAX_TOKENS_PROOF = 76800     # theorem proving (no answer)


def load_completed(output_path: Path) -> dict[int, dict]:
    """Load already-completed problem indices for resume support."""
    completed = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed[record["index"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


async def generate_one(
    client: AsyncOpenAI,
    model_id: str,
    row: dict,
    index: int,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    output_file,
    max_tokens: int,
    temperature: float,
    max_retries: int = 5,
):
    """Generate a single trace with exponential backoff retry."""
    async with semaphore:
        # Use higher budget and appropriate prompt for proof problems
        is_proof = row.get("answer") is None or str(row.get("answer", "")).strip() in ("", "None")
        if max_tokens is None:
            max_tokens = MAX_TOKENS_PROOF if is_proof else MAX_TOKENS_MATH
        if is_proof:
            prompt = f'Generate a rigorous proof to the following question: \n\n{row["problem"]}'
        else:
            prompt = f'{row["problem"]}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.'
        messages = [{"role": "user", "content": prompt}]

        last_error = None
        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=model_id,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = await client.chat.completions.create(**kwargs)
                choice = response.choices[0]

                # Extract reasoning/thinking if present
                reasoning = None
                content = choice.message.content
                if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                    reasoning = choice.message.reasoning_content
                elif hasattr(choice.message, "reasoning") and choice.message.reasoning:
                    reasoning = choice.message.reasoning

                result = {
                    "index": index,
                    "problem": row["problem"],
                    "answer": row["answer"],
                    "source": row["source"],
                    "mean_reward": row["mean_reward"],
                    "response": content,
                    "reasoning": reasoning,
                    "finish_reason": choice.finish_reason,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    if response.usage
                    else None,
                }

                async with write_lock:
                    output_file.write(json.dumps(result) + "\n")
                    output_file.flush()

                return result

            except Exception as e:
                last_error = e
                wait = min(2**attempt * 2, 60)
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)

        # All retries exhausted — write error record
        result = {
            "index": index,
            "problem": row["problem"],
            "answer": row["answer"],
            "response": None,
            "error": str(last_error),
            "error_type": type(last_error).__name__,
        }
        async with write_lock:
            output_file.write(json.dumps(result) + "\n")
            output_file.flush()

        return result


async def main():
    parser = argparse.ArgumentParser(description="Generate traces via OpenRouter")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-235b-thinking",
        choices=list(MODELS.keys()),
        help="Model alias",
    )
    parser.add_argument("--output-dir", type=str, default="traces")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max tokens (default: 32k math, 75k proofs)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: per-model, 0.6 for most)")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    model_id = model_cfg["model_id"]
    # Use CLI override > per-model config > default 0.6
    if args.temperature is not None:
        temperature = args.temperature
    elif "temperature" in model_cfg:
        temperature = model_cfg["temperature"]
    else:
        temperature = 0.6
    output_path = Path(args.output_dir) / f"{args.model}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ────────────────────────────────────────────────────
    print("Loading GVF-3k dataset...")
    ds = load_dataset("haoranli-ml/GVF-3k", split="train")
    total = len(ds)

    start = args.start
    end = args.end if args.end is not None else total
    end = min(end, total)

    # ── Resume support ──────────────────────────────────────────────────
    completed = load_completed(output_path)
    remaining = [
        (i, ds[i]) for i in range(start, end) if i not in completed
    ]
    n_errors = sum(1 for r in completed.values() if r.get("response") is None)

    print(f"Model:       {model_id}")
    print(f"Range:       [{start}, {end})")
    print(f"Total:       {end - start}")
    print(f"Completed:   {len(completed)} ({n_errors} errors)")
    print(f"Remaining:   {len(remaining)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output:      {output_path}")
    print()

    if not remaining:
        print("All problems already completed!")
        return

    # ── Setup client ────────────────────────────────────────────────────
    api_key_env = model_cfg["api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} environment variable not set")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=model_cfg["base_url"],
        max_retries=0,  # We handle retries ourselves
        timeout=3600,  # 1 hour
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    t0 = time.time()
    with open(output_path, "a") as f:
        tasks = [
            generate_one(
                client,
                model_id,
                row,
                idx,
                semaphore,
                write_lock,
                f,
                args.max_tokens,
                temperature,
            )
            for idx, row in remaining
        ]

        results = []
        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Generating ({args.model})",
        ):
            result = await coro
            results.append(result)

    elapsed = time.time() - t0
    n_success = sum(1 for r in results if r.get("response") is not None)
    n_fail = len(results) - n_success

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Success: {n_success}")
    print(f"  Failed:  {n_fail}")
    print(f"  Output:  {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
