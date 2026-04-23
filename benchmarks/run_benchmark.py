#!/usr/bin/env python3
"""Benchmark mlxz vs mlx-lm inference performance.

Measures TTFT (time-to-first-token), decode throughput (tok/s), and total
latency across a matrix of prompt sizes and max_tokens values.  Results are
saved as JSON and printed as a formatted comparison table.

Usage:
    # Start mlxz server first:
    mlxz serve mlx-community/Llama-3.1-8B-Instruct-4bit --port 8321

    # Then run:
    python benchmarks/run_benchmark.py \
        --model mlx-community/Llama-3.1-8B-Instruct-4bit \
        --mlxz-url http://127.0.0.1:8321 \
        --output benchmarks/results/

    # Skip mlx-lm direct comparison (mlxz-only):
    python benchmarks/run_benchmark.py \
        --model mlx-community/Llama-3.1-8B-Instruct-4bit \
        --skip-mlx-lm

    # Custom matrix:
    python benchmarks/run_benchmark.py \
        --model mlx-community/Llama-3.1-8B-Instruct-4bit \
        --prompt-tokens 64 256 1024 4096 \
        --max-tokens 32 128 512 \
        --runs 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import httpx


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    system: str  # "mlxz" or "mlx-lm"
    model: str
    prompt_tokens: int
    max_tokens: int
    completion_tokens: int
    ttft_ms: float
    decode_tps: float
    total_latency_ms: float
    timestamp: str


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

_PADDING = [
    "Explain", "the", "fascinating", "history", "of", "ancient",
    "civilizations", "including", "their", "remarkable", "achievements",
    "in", "architecture", "mathematics", "astronomy", "philosophy",
    "literature", "agriculture", "trade", "governance", "and", "culture",
    "across", "different", "continents", "and", "eras", "with", "specific",
    "examples", "from", "Mesopotamia", "Egypt", "Greece", "Rome", "China",
    "India", "and", "Mesoamerica", "highlighting", "key", "innovations",
]


def generate_prompt(target_tokens: int) -> str:
    """Generate a prompt of approximately *target_tokens* length.

    Each whitespace-separated word is roughly one token for most BPE
    tokenizers.  The actual count will vary by model but is close enough
    for benchmarking purposes.
    """
    words: list[str] = []
    for i in range(target_tokens):
        words.append(_PADDING[i % len(_PADDING)])
    return " ".join(words)


# ---------------------------------------------------------------------------
# mlxz benchmark (HTTP streaming)
# ---------------------------------------------------------------------------


def benchmark_mlxz(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> BenchmarkResult:
    """Benchmark mlxz via streaming HTTP to get accurate TTFT."""
    t0 = time.perf_counter()
    first_token_time: float | None = None
    content_chunks = 0
    completion_tokens_from_usage = 0
    prompt_token_count = 0

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": True,
                "temperature": 0,
                "seed": 42,
            },
            headers={"Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    content_chunks += 1
                # Usage may appear in the final chunk
                if chunk.get("usage"):
                    prompt_token_count = chunk["usage"].get(
                        "prompt_tokens", prompt_token_count
                    )
                    completion_tokens_from_usage = chunk["usage"].get(
                        "completion_tokens", completion_tokens_from_usage
                    )

    total_time = time.perf_counter() - t0
    total_tokens = completion_tokens_from_usage or content_chunks
    ttft = (
        (first_token_time - t0) * 1000
        if first_token_time is not None
        else total_time * 1000
    )
    decode_time = total_time - (ttft / 1000)
    decode_tps = total_tokens / decode_time if decode_time > 0 else 0.0

    return BenchmarkResult(
        system="mlxz",
        model=model,
        prompt_tokens=prompt_token_count,
        max_tokens=max_tokens,
        completion_tokens=total_tokens,
        ttft_ms=round(ttft, 2),
        decode_tps=round(decode_tps, 2),
        total_latency_ms=round(total_time * 1000, 2),
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# mlx-lm direct benchmark
# ---------------------------------------------------------------------------


def benchmark_mlx_lm(
    model_path: str,
    prompt: str,
    max_tokens: int,
    *,
    _cache: dict[str, object] = {},
) -> BenchmarkResult:
    """Benchmark mlx-lm by calling its generate() directly.

    The model/tokenizer pair is cached across calls so loading cost is
    paid only once.
    """
    import mlx_lm  # noqa: PLC0415 — deferred so --skip-mlx-lm works without the dep

    # Load model once
    if model_path not in _cache:
        model, tokenizer = mlx_lm.load(model_path)
        _cache[model_path] = (model, tokenizer)
        # Warm-up pass
        mlx_lm.generate(model, tokenizer, prompt="Hi", max_tokens=1, verbose=False)

    model, tokenizer = _cache[model_path]

    # Determine prompt token count
    messages = [{"role": "user", "content": prompt}]
    template_result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if isinstance(template_result, dict):
        prompt_ids = template_result["input_ids"]
    elif hasattr(template_result, "input_ids"):
        prompt_ids = template_result.input_ids
    else:
        prompt_ids = list(template_result)

    t0 = time.perf_counter()
    first_token_time: float | None = None
    output_parts: list[str] = []
    output_tokens = 0
    for step in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        if step.text:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            output_parts.append(step.text)
            output_tokens += 1
    total_time = time.perf_counter() - t0
    ttft_ms = (
        (first_token_time - t0) * 1000
        if first_token_time is not None
        else total_time * 1000
    )
    decode_time = total_time - (ttft_ms / 1000)
    decode_tps = output_tokens / decode_time if decode_time > 0 else 0.0

    return BenchmarkResult(
        system="mlx-lm",
        model=model_path,
        prompt_tokens=len(prompt_ids),
        max_tokens=max_tokens,
        completion_tokens=output_tokens,
        ttft_ms=round(ttft_ms, 2),
        decode_tps=round(decode_tps, 2),
        total_latency_ms=round(total_time * 1000, 2),
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_comparison_table(results: list[BenchmarkResult]) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 100)
    print(
        f"{'System':<10} {'Prompt':<8} {'MaxTok':<8} {'CompTok':<8} "
        f"{'TTFT(ms)':<12} {'Decode(t/s)':<12} {'Total(ms)':<12}"
    )
    print("-" * 100)
    for r in sorted(
        results, key=lambda x: (x.prompt_tokens, x.max_tokens, x.system)
    ):
        print(
            f"{r.system:<10} {r.prompt_tokens:<8} {r.max_tokens:<8} "
            f"{r.completion_tokens:<8} {r.ttft_ms:<12.1f} "
            f"{r.decode_tps:<12.1f} {r.total_latency_ms:<12.1f}"
        )
    print("=" * 100)

    # Speedup summary
    mlxz_results = [r for r in results if r.system == "mlxz"]
    mlxlm_results = [r for r in results if r.system == "mlx-lm"]
    if not mlxlm_results:
        return

    print("\n--- Speedup Summary (mlxz vs mlx-lm) ---")
    for mr in mlxz_results:
        matching = [
            r
            for r in mlxlm_results
            if r.prompt_tokens == mr.prompt_tokens
            and r.max_tokens == mr.max_tokens
        ]
        if matching:
            m = matching[0]
            tps_ratio = (
                mr.decode_tps / m.decode_tps if m.decode_tps > 0 else float("inf")
            )
            ttft_ratio = (
                m.ttft_ms / mr.ttft_ms if mr.ttft_ms > 0 else float("inf")
            )
            latency_ratio = (
                m.total_latency_ms / mr.total_latency_ms
                if mr.total_latency_ms > 0
                else float("inf")
            )
            print(
                f"  Prompt={mr.prompt_tokens:>5}, MaxTok={mr.max_tokens:>4}: "
                f"Decode {tps_ratio:.2f}x, "
                f"TTFT {ttft_ratio:.2f}x, "
                f"Latency {latency_ratio:.2f}x"
            )


def _median_result(runs: list[BenchmarkResult]) -> BenchmarkResult:
    """Pick the median run by decode_tps."""
    runs.sort(key=lambda r: r.decode_tps)
    return runs[len(runs) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark mlxz vs mlx-lm inference performance"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.1-8B-Instruct-4bit",
        help="HuggingFace model repo ID (default: %(default)s)",
    )
    parser.add_argument(
        "--mlxz-url",
        default="http://127.0.0.1:8321",
        help="Base URL of the running mlxz server (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/",
        help="Directory to save result JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=[32, 128, 512],
        metavar="N",
        help="Max completion tokens to test (default: 32 128 512)",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[64, 256, 1024, 4096],
        metavar="N",
        help="Approximate prompt sizes to test (default: 64 256 1024 4096)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Runs per configuration; median is kept (default: %(default)s)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when baseline regressions are detected (default: true)",
    )
    parser.add_argument(
        "--skip-mlx-lm",
        action="store_true",
        help="Skip mlx-lm direct benchmarks (mlxz-only mode)",
    )
    parser.add_argument(
        "--skip-mlxz",
        action="store_true",
        help="Skip mlxz benchmarks (mlx-lm-only mode)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[BenchmarkResult] = []
    total_configs = len(args.prompt_tokens) * len(args.max_tokens)
    config_idx = 0

    for prompt_tok in args.prompt_tokens:
        prompt = generate_prompt(prompt_tok)
        for max_tok in args.max_tokens:
            config_idx += 1
            print(
                f"\n[{config_idx}/{total_configs}] "
                f"prompt~{prompt_tok} tokens, max_tokens={max_tok}"
            )

            # --- mlxz ---
            if not args.skip_mlxz:
                print(f"  mlxz ({args.runs} runs) ...", end="", flush=True)
                mlxz_runs: list[BenchmarkResult] = []
                for run_i in range(args.runs):
                    try:
                        result = benchmark_mlxz(
                            args.mlxz_url, args.model, prompt, max_tok
                        )
                        mlxz_runs.append(result)
                        print(
                            f"\n    run {run_i + 1}: "
                            f"{result.decode_tps:.1f} tok/s, "
                            f"TTFT={result.ttft_ms:.1f}ms",
                            end="",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"\n    run {run_i + 1}: FAILED - {exc}", end="")
                print()
                if mlxz_runs:
                    all_results.append(_median_result(mlxz_runs))

            # --- mlx-lm ---
            if not args.skip_mlx_lm:
                print(f"  mlx-lm ({args.runs} runs) ...", end="", flush=True)
                mlxlm_runs: list[BenchmarkResult] = []
                for run_i in range(args.runs):
                    try:
                        result = benchmark_mlx_lm(args.model, prompt, max_tok)
                        mlxlm_runs.append(result)
                        print(
                            f"\n    run {run_i + 1}: "
                            f"{result.decode_tps:.1f} tok/s, "
                            f"Total={result.total_latency_ms:.1f}ms",
                            end="",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"\n    run {run_i + 1}: FAILED - {exc}", end="")
                print()
                if mlxlm_runs:
                    all_results.append(_median_result(mlxlm_runs))

    if not all_results:
        print("\nNo successful benchmark runs. Exiting.")
        sys.exit(1)

    # Print comparison table
    print_comparison_table(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Save baseline if none exists
    baseline_file = Path("benchmarks/baseline.json")
    if not baseline_file.exists():
        with open(baseline_file, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"Baseline saved to {baseline_file}")
    else:
        # Compare against baseline
        regressions = _compare_baseline(baseline_file, all_results)
        if regressions > 0 and args.fail_on_regression:
            print("\nFailing due to benchmark regressions.")
            sys.exit(1)


def _compare_baseline(
    baseline_path: Path, current: list[BenchmarkResult]
) -> int:
    """Compare current results against a saved baseline and flag regressions."""
    with open(baseline_path) as f:
        baseline_data = json.load(f)

    print("\n--- Regression Check vs Baseline ---")
    regressions = 0
    for br in baseline_data:
        matching = [
            r
            for r in current
            if r.system == br["system"]
            and r.model == br["model"]
            and r.prompt_tokens == br["prompt_tokens"]
            and r.max_tokens == br["max_tokens"]
        ]
        if not matching:
            continue
        cur = matching[0]
        baseline_tps = br["decode_tps"]
        if baseline_tps <= 0:
            continue
        ratio = cur.decode_tps / baseline_tps
        status = "OK" if ratio >= 0.90 else "REGRESSION"
        if status == "REGRESSION":
            regressions += 1
        print(
            f"  {cur.system} prompt={cur.prompt_tokens} max={cur.max_tokens}: "
            f"{cur.decode_tps:.1f} vs {baseline_tps:.1f} tok/s "
            f"({ratio:.2f}x) [{status}]"
        )

    if regressions:
        print(f"\n  WARNING: {regressions} regression(s) detected (>10% slower)")
    else:
        print("\n  All results within 10% of baseline.")
    return regressions


if __name__ == "__main__":
    main()
