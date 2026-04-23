#!/usr/bin/env python3
"""Canonical agent-style benchmark with repeated shared system prompt."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx


SYSTEM_PROMPT = (
    "You are a precise coding assistant. Keep answers concise, factual, and actionable. "
    "When uncertain, state assumptions explicitly."
)


async def _run_one(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    user_prompt: str,
    max_tokens: int,
) -> dict:
    t0 = time.perf_counter()
    first_token_time: float | None = None
    completion_tokens = 0
    completion_tokens_from_usage = 0

    async with client.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0,
            "seed": 42,
        },
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk["choices"][0].get("delta", {}).get("content"):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                completion_tokens += 1
            if chunk.get("usage"):
                completion_tokens_from_usage = chunk["usage"].get(
                    "completion_tokens", completion_tokens_from_usage
                )

    total = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else total
    decode = total - ttft
    total_completion_tokens = completion_tokens_from_usage or completion_tokens
    return {
        "ttft_ms": ttft * 1000.0,
        "decode_tps": total_completion_tokens / decode if decode > 0 else 0.0,
        "total_ms": total * 1000.0,
        "completion_tokens": total_completion_tokens,
    }


async def main_async(args) -> None:
    prompts = [
        "Explain the difference between a mutex and semaphore.",
        "Write a Python function for topological sort.",
        "Summarize HTTP caching headers for APIs.",
        "List common causes of flaky CI tests.",
        "Show a safe SQL parameterization example in Python.",
        "Explain why backpressure matters in streaming APIs.",
        "Describe optimistic vs pessimistic locking.",
        "Give a concise incident response checklist.",
    ]
    async with httpx.AsyncClient(timeout=180) as client:
        tasks = [
            _run_one(
                client,
                args.mlxz_url,
                args.model,
                prompts[i % len(prompts)],
                args.max_tokens,
            )
            for i in range(args.requests)
        ]
        started = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - started

    ttfts = [r["ttft_ms"] for r in results]
    tps = [r["decode_tps"] for r in results]
    total_tokens = sum(r["completion_tokens"] for r in results)

    print("\nAgent workload summary")
    print(f"requests: {args.requests}")
    print(f"shared system prompt tokens: reused across all requests")
    print(f"median TTFT: {statistics.median(ttfts):.1f} ms")
    print(f"p95 TTFT: {sorted(ttfts)[int(len(ttfts) * 0.95) - 1]:.1f} ms")
    print(f"median decode throughput: {statistics.median(tps):.2f} tok/s")
    print(f"aggregate throughput: {total_tokens / wall:.2f} tok/s")
    print(f"wall time: {wall:.2f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical agent-style workload benchmark")
    parser.add_argument("--mlxz-url", default="http://127.0.0.1:8321")
    parser.add_argument("--model", default="mlx-community/Llama-3.1-8B-Instruct-4bit")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
