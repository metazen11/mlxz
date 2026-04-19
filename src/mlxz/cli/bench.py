"""mlxz bench — benchmark and regression detection."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
import httpx
import structlog

logger = structlog.get_logger()


def bench(
    regression: Annotated[bool, typer.Option(help="Compare against baseline")] = False,
    baseline: Annotated[Path | None, typer.Option(help="Baseline JSON file")] = None,
    url: Annotated[str, typer.Option(help="mlxz server URL")] = "http://127.0.0.1:8000",
    max_tokens: Annotated[int, typer.Option(help="Tokens to generate")] = 64,
    runs: Annotated[int, typer.Option(help="Runs per config")] = 3,
    matrix: Annotated[bool, typer.Option(help="Run full benchmark matrix")] = False,
) -> None:
    """Run benchmarks and optionally check for regressions."""

    # Check server is running
    try:
        resp = httpx.get(f"{url}/health/ready", timeout=5)
        if resp.status_code != 200:
            typer.echo("Server not ready. Start with: mlxz serve <model>", err=True)
            raise typer.Exit(1)
    except httpx.ConnectError:
        typer.echo(f"Cannot connect to {url}. Start the server first.", err=True)
        raise typer.Exit(1)

    # Get model name
    models = httpx.get(f"{url}/v1/models", timeout=5).json()
    model_name = models["data"][0]["id"] if models.get("data") else "unknown"
    typer.echo(f"Benchmarking model: {model_name}")

    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "What are the key differences between TCP and UDP?",
    ]

    results = []
    for prompt in prompts:
        tps_values = []
        ttft_values = []

        for run in range(runs):
            t0 = time.perf_counter()
            first_token_time = None
            token_count = 0

            with httpx.Client(timeout=120) as client:
                with client.stream("POST", f"{url}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "stream": True,
                        "temperature": 0,
                        "seed": 42,
                    },
                ) as resp:
                    for line in resp.iter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        chunk = json.loads(line[6:])
                        if chunk["choices"][0]["delta"].get("content"):
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1

            total = time.perf_counter() - t0
            ttft = (first_token_time - t0) * 1000 if first_token_time else 0
            decode_time = total - (ttft / 1000)
            tps = token_count / decode_time if decode_time > 0 else 0

            tps_values.append(tps)
            ttft_values.append(ttft)
            typer.echo(f"  Run {run+1}: {tps:.1f} tok/s, TTFT={ttft:.0f}ms")

        # Median
        tps_values.sort()
        ttft_values.sort()
        median_tps = tps_values[len(tps_values) // 2]
        median_ttft = ttft_values[len(ttft_values) // 2]

        results.append({
            "prompt": prompt[:50],
            "max_tokens": max_tokens,
            "median_tps": round(median_tps, 2),
            "median_ttft_ms": round(median_ttft, 2),
            "model": model_name,
        })

    # Print results table
    typer.echo("\n" + "=" * 70)
    typer.echo(f"{'Prompt':<50} {'tok/s':>8} {'TTFT':>8}")
    typer.echo("-" * 70)
    for r in results:
        typer.echo(f"{r['prompt']:<50} {r['median_tps']:>8.1f} {r['median_ttft_ms']:>7.0f}ms")
    typer.echo("=" * 70)

    # Regression check
    if regression:
        baseline_path = baseline or Path("benchmarks/baseline.json")
        if not baseline_path.exists():
            typer.echo(f"\nNo baseline at {baseline_path}. Saving current as baseline.")
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(results, indent=2))
            raise typer.Exit(0)

        baseline_data = json.loads(baseline_path.read_text())
        typer.echo(f"\nRegression check against {baseline_path}:")

        failed = False
        for current, base in zip(results, baseline_data):
            if isinstance(base, dict) and "median_tps" in base:
                base_tps = base["median_tps"]
                delta = (current["median_tps"] - base_tps) / base_tps * 100
                status = "PASS" if delta >= -5 else "FAIL"
                if status == "FAIL":
                    failed = True
                typer.echo(f"  [{status}] {current['prompt']}: "
                           f"{current['median_tps']:.1f} vs {base_tps:.1f} tok/s ({delta:+.1f}%)")

        if failed:
            typer.echo("\nRegression detected (>5% slowdown)!", err=True)
            raise typer.Exit(1)
        else:
            typer.echo("\nAll benchmarks within tolerance.")
