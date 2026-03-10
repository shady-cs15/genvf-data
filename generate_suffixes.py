#!/usr/bin/env python3
"""Generate suffixes by continuing from truncated prefixes.

For each prefix, constructs a prompt with the original problem and the prefix
as a partial assistant response, then lets the model continue generating.

Usage:
    python generate_suffixes_from_gemini.py \
    --model gemini-3-flash \
    --hf-dataset haoranli-ml/genvf-prefixes-filtered \
    --hf-split train_replaced_none_prefix \
    --num-suffixes 8 \
    --concurrency 16 \
    --output-dir /path/to/suffixes

    python generate_suffixes_from_gemini.py --model gemini-3-flash --start 0 --end 1
    python generate_suffixes_from_gemini.py --model gemini-3-pro --num-suffixes 8
    python generate_suffixes_from_gemini.py --model gemini-3-flash --start 0 --end 100 --concurrency 8
"""

import asyncio
import json
import os
import argparse
import time
import re
import random
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    "qwen3-235b-thinking": {
        "model_id": "qwen/qwen3-235b-a22b-thinking-2507",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "temperature": 0.6,
    },
    "gemini-3-flash": {
        "model_id": "gemini-3-flash-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "temperature": None,  # Use Gemini's server-side default unless overridden via CLI
    },
    "gemini-3-pro": {
        "model_id": "gemini-3-pro-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "temperature": None,  # Use Gemini's server-side default unless overridden via CLI
    },
    "gpt5-mini": {
        "model_id": "gpt-5-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": None,
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

# Models that use explicit reasoning tags
THINK_TAG_MODELS = {"qwen3-235b-thinking", "gemini-3-flash", "gemini-3-pro"}
MODEL_THINK_TAGS = {
    "qwen3-235b-thinking": "think",
    "gemini-3-flash": "thought",
    "gemini-3-pro": "thought",
}

# Models that use multi-turn continuation (no prefill support)
# Prefix is placed as a prior assistant turn, then a follow-up user message asks to continue.
MULTI_TURN_MODELS = {"qwen3.5-27b", "qwen3.5-35b-a3b", "gpt5-mini"}

# Models where suffix generation is not supported
UNSUPPORTED_MODELS = {"gpt5-mini"}

# Pool of models for --model random
RANDOM_POOL = [
    "qwen3-235b-thinking",
    "gpt5-mini",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "gemini-3-flash",
    "gemini-3-pro",
]

# Models that write a dummy placeholder instead of calling the API
DUMMY_MODELS = {"gemini-3-flash", "gemini-3-pro"}

# ── Token budgets by problem type ────────────────────────────────────────────
# INITIAL_MAX_TOKENS_MATH = 32768
# INITIAL_MAX_TOKENS_PROOF = 76800
# MAX_BUDGET_ESCALATIONS = 3  # how many times to double the budget on truncation
# ABSOLUTE_MAX_TOKENS = 229376

INITIAL_MAX_TOKENS_MATH = 50000
INITIAL_MAX_TOKENS_PROOF = 120000
MAX_BUDGET_ESCALATIONS = 3  # how many times to double the budget on truncation
ABSOLUTE_MAX_TOKENS = 229376


def classify_error(error_type: str | None, error_msg: str | None) -> str:
    """Classify errors for live stats reporting."""
    text = f"{error_type or ''} {error_msg or ''}".lower()
    if "429" in text or "ratelimit" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limit"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    return "other"


async def periodic_stats_reporter(
    stats: dict[str, int],
    total_tasks: int,
    done_event: asyncio.Event,
    interval: int = 30,
):
    """Print progress and counters every interval seconds while running."""
    while True:
        try:
            await asyncio.wait_for(done_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            completed = (
                stats["stop"]
                + stats["length"]
                + stats["other_finish"]
                + stats["rate_limit"]
                + stats["timeout"]
                + stats["other"]
            )
            print(
                "[stats] "
                f"{completed}/{total_tasks} done | "
                f"stop={stats['stop']} | "
                f"length={stats['length']} | "
                f"429={stats['rate_limit']} | "
                f"timeout={stats['timeout']} | "
                f"other_err={stats['other']}"
            )


def load_prefix_rows(path: Path) -> list[dict]:
    """Load prefix JSONL records as a row list (preserve every row)."""
    rows = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "index" not in record:
                        continue
                    record = dict(record)
                    record["row_id"] = len(rows)
                    rows.append(record)
                except (json.JSONDecodeError, KeyError):
                    continue
    return rows


def load_prefix_rows_from_hf(dataset_name: str, split: str) -> list[dict]:
    """Load HF dataset records as a row list (preserve every row).

    Assumes column names are the same as local jsonl keys.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Hugging Face mode requires `datasets`. Install with: pip install datasets"
        ) from e

    ds = load_dataset(dataset_name, split=split)
    rows = []

    for row in ds:
        idx = row.get("index")
        if idx is None:
            continue
        record = dict(row)
        record["index"] = int(idx)
        record["row_id"] = len(rows)
        rows.append(record)

    return rows


def load_records_by_index(path: Path) -> dict[int, dict]:
    """Load JSONL records keyed by index."""
    records = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records[record["index"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue
    return records


def completed_key(record: dict) -> tuple[str, int] | None:
    """Return resume key; prefer row_id so duplicate indices are independent."""
    if "row_id" in record:
        try:
            return ("row_id", int(record["row_id"]))
        except (TypeError, ValueError):
            return None
    if "index" in record:
        try:
            return ("index", int(record["index"]))
        except (TypeError, ValueError):
            return None
    return None


def load_completed_suffixes(path: Path) -> dict[tuple[str, int], list[dict]]:
    """Load completed suffixes grouped by resume key."""
    completed = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = completed_key(record)
                    if key is None:
                        continue
                    if key not in completed:
                        completed[key] = []
                    completed[key].append(record)
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def extract_original_suffix(prefix_record: dict, trace_record: dict, model: str) -> dict | None:
    """Try to extract suffix 0 from the original trace.

    Returns a suffix record if the original trace was complete (finish_reason=stop
    and response is not None), otherwise returns None (needs regeneration).
    """
    if trace_record.get("finish_reason") != "stop":
        return None
    if trace_record.get("response") is None:
        return None

    prefix = prefix_record["prefix"]

    # Extract the suffix portion of the reasoning (everything after the prefix)
    if model in THINK_TAG_MODELS:
        full_reasoning = trace_record.get("reasoning") or ""
        # Find where the prefix ends in the full reasoning
        prefix_pos = full_reasoning.find(prefix)
        if prefix_pos == -1:
            return None
        suffix_reasoning = full_reasoning[prefix_pos + len(prefix):]
    else:
        suffix_reasoning = None

    return {
        "index": prefix_record["index"],
        "row_id": prefix_record["row_id"],
        "suffix_num": 0,
        "problem": prefix_record["problem"],
        "answer": prefix_record.get("answer"),
        "source": prefix_record.get("source"),
        "mean_reward": prefix_record.get("mean_reward"),
        "prefix": prefix,
        "prefix_type": prefix_record.get("prefix_type"),
        "prefix_end_index": prefix_record.get("prefix_end_index"),
        "num_thoughts": prefix_record.get("num_thoughts"),
        "suffix_response": trace_record.get("response"),
        "suffix_reasoning": suffix_reasoning,
        "finish_reason": "stop",
        "budget_used": 0,
        "escalation": 0,
        "prefix_model": prefix_record.get("model"),
        "suffix_model": trace_record.get("model"),
        "model": trace_record.get("model"),
        "usage": trace_record.get("usage"),
        "from_original_trace": True,
    }


def build_messages(prefix_record: dict, model: str) -> list[dict]:
    """Build the message list for suffix generation.

    For thinking models, prefill the assistant's reasoning block with the prefix.
    """
    problem = prefix_record["problem"]
    prefix = prefix_record["prefix"]
    answer = prefix_record.get("answer")

    # Use the same prompt format as generate_traces.py
    is_proof = answer is None or str(answer).strip() in ("", "None")
    if is_proof:
        user_prompt = f"Generate a rigorous proof to the following question: \n\n{problem}"
    else:
        user_prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    if model in MULTI_TURN_MODELS:
        # These models don't support assistant prefill continuation.
        # Use multi-turn: place prefix as a prior assistant turn, then an empty
        # user turn so the model continues naturally without meta-commentary.
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": prefix},
            {"role": "user", "content": ""},
        ]
    elif model in THINK_TAG_MODELS:
        # Prefill assistant with an open reasoning tag + prefix reasoning.
        # The model then continues reasoning and writes the answer.
        think_tag = MODEL_THINK_TAGS[model]
        assistant_prefill = f"<{think_tag}>\n{prefix}"

        # OpenRouter/DeepSeek style "prefix": true can help for some models (e.g., Qwen),
        # while Gemini works more reliably with plain assistant prefill.
        assistant_msg = {"role": "assistant", "content": assistant_prefill}
        if model == "qwen3-235b-thinking":
            assistant_msg["prefix"] = True

        return [
            {"role": "user", "content": user_prompt},
            assistant_msg,
        ]
    else:
        raise NotImplementedError(
            f"Suffix generation not supported for model '{model}': "
            f"internal chain-of-thought is not exposed by the API, "
            f"so prefix continuation is not possible."
        )


def extract_reasoning_from_message(message, content: str | None) -> str | None:
    """Extract reasoning text from model response, with tag-based fallback."""
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return message.reasoning_content
    if hasattr(message, "reasoning") and message.reasoning:
        return message.reasoning

    if isinstance(content, str) and content:
        for tag in ("thought", "think"):
            m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", content, flags=re.DOTALL)
            if m:
                extracted = m.group(1).strip()
                if extracted:
                    return extracted
    return None


def extract_thought_tags_from_response(content: str | None) -> str | None:
    """Extract all <thought>...</thought> blocks from response text."""
    if not isinstance(content, str) or not content:
        return None
    matches = re.findall(r"<thought>\s*(.*?)\s*</thought>", content, flags=re.DOTALL | re.IGNORECASE)
    chunks = [m.strip() for m in matches if isinstance(m, str) and m.strip()]
    if not chunks:
        return None
    return "\n\n".join(chunks)


async def write_dummy_suffix(
    prefix_record: dict,
    suffix_num: int,
    model: str,
    write_lock: asyncio.Lock,
    output_file,
):
    """Write a placeholder suffix record for models that need manual generation later."""
    model_cfg = MODELS[model]
    result = {
        "index": prefix_record["index"],
        "row_id": prefix_record["row_id"],
        "suffix_num": suffix_num,
        "problem": prefix_record["problem"],
        "answer": prefix_record.get("answer"),
        "source": prefix_record.get("source"),
        "mean_reward": prefix_record.get("mean_reward"),
        "prefix": prefix_record["prefix"],
        "prefix_type": prefix_record.get("prefix_type"),
        "prefix_end_index": prefix_record.get("prefix_end_index"),
        "num_thoughts": prefix_record.get("num_thoughts"),
        "suffix_response": None,
        "suffix_reasoning": None,
        "finish_reason": None,
        "budget_used": None,
        "escalation": None,
        "prefix_model": prefix_record.get("model"),
        "suffix_model": model_cfg["model_id"],
        "model": model_cfg["model_id"],
        "usage": None,
        "pending": True,
        "pending_model": model,
    }
    async with write_lock:
        output_file.write(json.dumps(result) + "\n")
        output_file.flush()
    return result


async def generate_one_suffix(
    client: AsyncOpenAI,
    model_id: str,
    model: str,
    prefix_record: dict,
    suffix_num: int,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    output_file,
    temperature: float,
    max_retries: int = 5,
):
    """Generate a single suffix with budget escalation on truncation."""
    async with semaphore:
        messages = build_messages(prefix_record, model)
        index = prefix_record["index"]
        row_id = prefix_record["row_id"]
        answer = prefix_record.get("answer")
        is_proof = answer is None or str(answer).strip() in ("", "None")
        base_budget = INITIAL_MAX_TOKENS_PROOF if is_proof else INITIAL_MAX_TOKENS_MATH

        last_error = None
        last_result = None

        # Try with escalating budgets
        for escalation in range(MAX_BUDGET_ESCALATIONS + 1):
            budget = min(base_budget * (2 ** escalation), ABSOLUTE_MAX_TOKENS)

            for attempt in range(max_retries):
                try:
                    kwargs = dict(
                        model=model_id,
                        messages=messages,
                        max_completion_tokens=budget,
                    )
                    if temperature is not None:
                        kwargs["temperature"] = temperature

                    response = await client.chat.completions.create(**kwargs)
                    choice = response.choices[0]

                    # Extract reasoning/thinking if present
                    content = choice.message.content
                    reasoning = extract_reasoning_from_message(choice.message, content)
                    thought_reasoning = extract_thought_tags_from_response(content)
                    if thought_reasoning:
                        if isinstance(reasoning, str) and reasoning.strip():
                            if thought_reasoning not in reasoning:
                                reasoning = f"{reasoning}\n\n{thought_reasoning}"
                        else:
                            reasoning = thought_reasoning

                    result = {
                        "index": index,
                        "row_id": row_id,
                        "suffix_num": suffix_num,
                        "problem": prefix_record["problem"],
                        "answer": prefix_record.get("answer"),
                        "source": prefix_record.get("source"),
                        "mean_reward": prefix_record.get("mean_reward"),
                        "prefix": prefix_record["prefix"],
                        "prefix_type": prefix_record.get("prefix_type"),
                        "prefix_end_index": prefix_record.get("prefix_end_index"),
                        "num_thoughts": prefix_record.get("num_thoughts"),
                        "suffix_response": content,
                        "suffix_reasoning": reasoning,
                        "finish_reason": choice.finish_reason,
                        "budget_used": budget,
                        "escalation": escalation,
                        "prefix_model": prefix_record.get("model"),
                        "suffix_model": response.model,
                        "model": response.model,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                        if response.usage
                        else None,
                    }

                    # If finished, write and return
                    if choice.finish_reason == "stop":
                        async with write_lock:
                            output_file.write(json.dumps(result) + "\n")
                            output_file.flush()
                        return result

                    # Truncated — save as last_result, try higher budget
                    last_result = result
                    break  # break retry loop, escalate budget

                except Exception as e:
                    last_error = e
                    wait = min(2**attempt * 2, 60)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait)
            else:
                # All retries exhausted at this budget level — write error
                result = {
                    "index": index,
                    "row_id": row_id,
                    "suffix_num": suffix_num,
                    "problem": prefix_record["problem"],
                    "answer": prefix_record.get("answer"),
                    "prefix": prefix_record["prefix"],
                    "suffix_response": None,
                    "finish_reason": None,
                    "budget_used": budget,
                    "prefix_model": prefix_record.get("model"),
                    "suffix_model": model_id,
                    "error": str(last_error),
                    "error_type": type(last_error).__name__,
                }
                async with write_lock:
                    output_file.write(json.dumps(result) + "\n")
                    output_file.flush()
                return result

            # If we already hit absolute max, stop escalating
            if budget >= ABSOLUTE_MAX_TOKENS:
                break

        # Exhausted all budget escalations — write the truncated result
        if last_result:
            async with write_lock:
                output_file.write(json.dumps(last_result) + "\n")
                output_file.flush()
            return last_result

        # Shouldn't reach here, but just in case
        result = {
            "index": index,
            "row_id": row_id,
            "suffix_num": suffix_num,
            "problem": prefix_record["problem"],
            "answer": prefix_record.get("answer"),
            "prefix": prefix_record["prefix"],
            "suffix_response": None,
            "finish_reason": None,
            "budget_used": None,
            "prefix_model": prefix_record.get("model"),
            "suffix_model": model_id,
            "error": str(last_error) if last_error else "Unknown error",
            "error_type": type(last_error).__name__ if last_error else "Unknown",
        }
        async with write_lock:
            output_file.write(json.dumps(result) + "\n")
            output_file.flush()
        return result


async def main():
    parser = argparse.ArgumentParser(description="Generate suffixes from prefixes")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()) + ["random"],
        help="Model to use for suffix generation (use 'random' to sample from pool)",
    )
    parser.add_argument("--trace-dir", type=str, default="traces",
                        help="Unused (trace reuse disabled)")
    parser.add_argument("--prefix-dir", type=str, default="prefixes")
    parser.add_argument("--output-dir", type=str, default="suffixes")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="Optional HF dataset name/path for prefixes")
    parser.add_argument("--hf-split", type=str, default="train",
                        help="HF split (used with --hf-dataset)")
    parser.add_argument("--num-suffixes", "-n", type=int, default=1,
                        help="Number of suffixes to generate per prefix")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature")
    parser.add_argument("--start", type=int, default=0, help="Start row (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End row (exclusive)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (used with --model random)")
    args = parser.parse_args()

    model = args.model
    is_random = model == "random"

    if not is_random and model in UNSUPPORTED_MODELS:
        raise NotImplementedError(
            f"Suffix generation not supported for model '{model}': "
            f"internal chain-of-thought is not exposed by the API, "
            f"so prefix continuation is not possible."
        )

    if is_random:
        rng = random.Random(args.seed)
        output_path = Path(args.output_dir) / "random.jsonl"
    else:
        model_cfg = MODELS[model]
        model_id = model_cfg["model_id"]
        output_path = Path(args.output_dir) / f"{model}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Temperature: CLI > per-model > default 0.6
    def get_temperature(m: str) -> float | None:
        if args.temperature is not None:
            return args.temperature
        cfg = MODELS[m]
        if "temperature" in cfg:
            return cfg["temperature"]
        return 0.6

    # Resolve prefix path: for random mode, must use --hf-dataset or --prefix-dir with a specific file
    if is_random:
        prefix_path = Path(args.prefix_dir) / "merged_v0.jsonl"
    else:
        prefix_path = Path(args.prefix_dir) / f"{model}.jsonl"

    # Load prefixes as rows (no trace reuse)
    if args.hf_dataset:
        prefix_rows = load_prefix_rows_from_hf(args.hf_dataset, args.hf_split)
    else:
        if not prefix_path.exists():
            raise FileNotFoundError(f"Prefix file not found: {prefix_path}")
        prefix_rows = load_prefix_rows(prefix_path)

    if not prefix_rows:
        raise ValueError("No prefix rows loaded.")

    # Determine row range
    total_rows = len(prefix_rows)
    start = max(0, args.start)
    end = args.end if args.end is not None else total_rows
    end = min(end, total_rows)
    if end < start:
        end = start
    row_ids = list(range(start, end))

    # Filter out skipped prefixes (no content to continue from)
    row_ids = [rid for rid in row_ids if prefix_rows[rid].get("prefix") is not None]

    # Resume support: check which (index, suffix_num) pairs are done
    completed = load_completed_suffixes(output_path)

    # Build remaining tasks
    # For random mode, pre-assign a model to each (rid, suffix_num) deterministically
    task_models = {}  # (rid, suffix_num) -> model alias
    tasks_to_run = []
    for rid in row_ids:
        key = ("row_id", rid)
        existing = completed.get(key, [])
        existing_nums = {r["suffix_num"] for r in existing}
        for s in range(args.num_suffixes):
            if is_random:
                task_models[(rid, s)] = rng.choice(RANDOM_POOL)
            if s not in existing_nums:
                tasks_to_run.append((rid, s))

    n_total = len(row_ids) * args.num_suffixes
    n_done = n_total - len(tasks_to_run)

    print(f"Model:       {'random (mixed pool)' if is_random else MODELS[model]['model_id']}")
    if is_random:
        # Count model assignments for remaining tasks
        from collections import Counter
        task_model_counts = Counter(task_models[k] for k in tasks_to_run)
        print(f"Random pool: {RANDOM_POOL}")
        print(f"Seed:        {args.seed}")
        print(f"Model distribution (remaining): {dict(task_model_counts)}")
    if args.hf_dataset:
        print(f"Prefixes:    hf://{args.hf_dataset} ({len(prefix_rows)} rows, split={args.hf_split})")
    else:
        print(f"Prefixes:    {prefix_path} ({len(prefix_rows)} rows)")
    print(f"Row range:   [{start}, {end})")
    print(f"Rows:        {len(row_ids)} (with valid prefix)")
    print("Trace reuse: disabled")
    print(f"Suffixes/prefix: {args.num_suffixes}")
    print(f"Total jobs:  {n_total}")
    print(f"Completed:   {n_done}")
    print(f"Remaining:   {len(tasks_to_run)}")
    print(f"Concurrency: {args.concurrency}")
    if not is_random:
        print(f"Temperature: {get_temperature(model)}")
    print(f"Output:      {output_path}")
    print()

    if not tasks_to_run:
        print("All suffixes already generated!")
        return

    # Setup clients — one per unique (base_url, api_key_env) pair
    clients = {}
    if is_random:
        models_needed = set(task_models[k] for k in tasks_to_run) - DUMMY_MODELS
    else:
        models_needed = {model} - DUMMY_MODELS

    for m in models_needed:
        cfg = MODELS[m]
        client_key = (cfg["base_url"], cfg["api_key_env"])
        if client_key not in clients:
            api_key = os.environ.get(cfg["api_key_env"])
            if not api_key:
                raise RuntimeError(f"{cfg['api_key_env']} environment variable not set (needed for {m})")
            clients[client_key] = AsyncOpenAI(
                api_key=api_key,
                base_url=cfg["base_url"],
                max_retries=0,
                timeout=3600,
            )

    def get_client(m: str) -> AsyncOpenAI:
        cfg = MODELS[m]
        return clients[(cfg["base_url"], cfg["api_key_env"])]

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    stats = {
        "stop": 0,
        "length": 0,
        "other_finish": 0,
        "rate_limit": 0,
        "timeout": 0,
        "other": 0,
        "dummy": 0,
    }
    done_event = asyncio.Event()

    t0 = time.time()
    with open(output_path, "a") as f:
        coros = []
        for rid, suffix_num in tasks_to_run:
            task_model = task_models[(rid, suffix_num)] if is_random else model
            if task_model in DUMMY_MODELS:
                coros.append(
                    write_dummy_suffix(
                        prefix_rows[rid],
                        suffix_num,
                        task_model,
                        write_lock,
                        f,
                    )
                )
            else:
                task_cfg = MODELS[task_model]
                coros.append(
                    generate_one_suffix(
                        get_client(task_model),
                        task_cfg["model_id"],
                        task_model,
                        prefix_rows[rid],
                        suffix_num,
                        semaphore,
                        write_lock,
                        f,
                        get_temperature(task_model),
                    )
                )

        reporter_task = asyncio.create_task(
            periodic_stats_reporter(stats=stats, total_tasks=len(coros), done_event=done_event, interval=30)
        )

        results = []
        desc = "Suffixes (random)" if is_random else f"Suffixes ({model})"
        for coro in atqdm(
            asyncio.as_completed(coros),
            total=len(coros),
            desc=desc,
        ):
            result = await coro
            results.append(result)
            if result.get("pending"):
                stats["dummy"] += 1
            elif result.get("suffix_response") is not None:
                finish_reason = result.get("finish_reason")
                if finish_reason == "stop":
                    stats["stop"] += 1
                elif finish_reason == "length":
                    stats["length"] += 1
                else:
                    stats["other_finish"] += 1
            else:
                bucket = classify_error(result.get("error_type"), result.get("error"))
                stats[bucket] += 1

        done_event.set()
        await reporter_task

    elapsed = time.time() - t0
    n_stop = sum(1 for r in results if r.get("finish_reason") == "stop")
    n_length = sum(1 for r in results if r.get("finish_reason") == "length")
    n_dummy = sum(1 for r in results if r.get("pending"))
    n_error = sum(1 for r in results if r.get("suffix_response") is None and not r.get("pending"))
    n_other_finish = sum(
        1
        for r in results
        if r.get("suffix_response") is not None and r.get("finish_reason") not in {"stop", "length"}
    )
    n_rate_limit = sum(
        1
        for r in results
        if r.get("suffix_response") is None and not r.get("pending")
        and classify_error(r.get("error_type"), r.get("error")) == "rate_limit"
    )
    n_timeout = sum(
        1
        for r in results
        if r.get("suffix_response") is None and not r.get("pending")
        and classify_error(r.get("error_type"), r.get("error")) == "timeout"
    )
    n_other_error = n_error - n_rate_limit - n_timeout

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Finished (stop):    {n_stop}")
    print(f"  Truncated (length): {n_length}")
    if n_dummy:
        print(f"  Dummy (pending):    {n_dummy}")
    print(f"  Other finish:       {n_other_finish}")
    print(f"  Errors:             {n_error}")
    print(f"    Rate limit (429): {n_rate_limit}")
    print(f"    Timeout:          {n_timeout}")
    print(f"    Other errors:     {n_other_error}")
    print(f"  Output:             {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
