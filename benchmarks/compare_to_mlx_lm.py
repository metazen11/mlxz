#!/usr/bin/env python3
"""Quick A/B comparison: mlxz server vs mlx-lm on a single prompt.

A simpler alternative to run_benchmark.py when you want a fast side-by-side
result without the full matrix sweep.

Usage:
    # mlxz server must be running:
    mlxz serve mlx-community/Llama-3.1-8B-Instruct-4bit --port 8321

    python benchmarks/compare_to_mlx_lm.py \
        --model mlx-community/Llama-3.1-8B-Instruct-4bit

    # Custom prompt:
    python benchmarks/compare_to_mlx_lm.py \
        --model mlx-community/Llama-3.1-8B-Instruct-4bit \
        --prompt "Write a Python function that sorts a list using merge sort."

    # mlxz-only (no mlx-lm comparison):
    python benchmarks/compare_to_mlx_lm.py \
        --model mlx-community/Llama-3.1-8B-Instruct-4bit \
        --skip-mlx-lm
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

DEFAULT_PROMPT = (
    "Explain the key differences between TCP and UDP protocols, "
    "including use cases for each."
)


def run_mlxz(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Run a single streaming request against mlxz and return timing data."""
    t0 = time.perf_counter()
    first_token_time: float | None = None
    tokens: list[str] = []

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": True,
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
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens.append(content)

    total = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else total
    decode_time = total - ttft
    n_tokens = len(tokens)

    return {
        "system": "mlxz",
        "tokens": n_tokens,
        "ttft_ms": round(ttft * 1000, 1),
        "decode_tps": round(n_tokens / decode_time, 1) if decode_time > 0 else 0,
        "total_ms": round(total * 1000, 1),
        "output": "".join(tokens),
    }


def run_mlx_lm(model_path: str, prompt: str, max_tokens: int) -> dict:
    """Run mlx-lm generate() directly and return timing data."""
    import mlx_lm  # noqa: PLC0415

    model, tokenizer = mlx_lm.load(model_path)
    # Warm-up
    mlx_lm.generate(model, tokenizer, prompt="Hi", max_tokens=1, verbose=False)

    t0 = time.perf_counter()
    output = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
    )
    total = time.perf_counter() - t0

    n_tokens = len(tokenizer.encode(output))

    return {
        "system": "mlx-lm",
        "tokens": n_tokens,
        "ttft_ms": round(total * 1000 / max(n_tokens, 1), 1),
        "decode_tps": round(n_tokens / total, 1) if total > 0 else 0,
        "total_ms": round(total * 1000, 1),
        "output": output,
    }


def print_side_by_side(a: dict, b: dict | None) -> None:
    """Print a side-by-side comparison of two results."""
    width = 40
    print()
    print(f"{'=' * (width * 2 + 3)}")
    print(f"{'mlxz':^{width}} | {'mlx-lm':^{width}}")
    print(f"{'-' * width}-+-{'-' * width}")

    rows = [
        ("Tokens generated", "tokens"),
        ("TTFT (ms)", "ttft_ms"),
        ("Decode (tok/s)", "decode_tps"),
        ("Total latency (ms)", "total_ms"),
    ]
    for label, key in rows:
        a_val = str(a[key])
        b_val = str(b[key]) if b else "skipped"
        print(f"  {label + ':':<22} {a_val:<{width - 24}} | {b_val}")

    # Speedup
    if b:
        tps_ratio = a["decode_tps"] / b["decode_tps"] if b["decode_tps"] > 0 else 0
        ttft_ratio = b["ttft_ms"] / a["ttft_ms"] if a["ttft_ms"] > 0 else 0
        print(f"\n  Decode speedup: {tps_ratio:.2f}x")
        print(f"  TTFT speedup:   {ttft_ratio:.2f}x")

    print(f"{'=' * (width * 2 + 3)}")

    # Show truncated output
    print(f"\n--- mlxz output (first 200 chars) ---")
    print(a["output"][:200])
    if b:
        print(f"\n--- mlx-lm output (first 200 chars) ---")
        print(b["output"][:200])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick A/B: mlxz vs mlx-lm on a single prompt"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.1-8B-Instruct-4bit",
    )
    parser.add_argument("--mlxz-url", default="http://127.0.0.1:8321")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--skip-mlx-lm", action="store_true", help="Only run mlxz benchmark"
    )
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt[:80]}...")
    print(f"Max tokens: {args.max_tokens}")

    print("\nRunning mlxz ...")
    mlxz_result = run_mlxz(args.mlxz_url, args.model, args.prompt, args.max_tokens)

    mlxlm_result = None
    if not args.skip_mlx_lm:
        print("Running mlx-lm ...")
        mlxlm_result = run_mlx_lm(args.model, args.prompt, args.max_tokens)

    print_side_by_side(mlxz_result, mlxlm_result)


if __name__ == "__main__":
    main()
