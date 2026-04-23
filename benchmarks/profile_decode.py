#!/usr/bin/env python3
"""Profile decode-path cost split for MLX models.

The script measures where decode time goes on a greedy decode workload:
- total model forward time
- cache.update_and_fetch time
- scaled dot-product attention time
- sampler / token selection time

This is meant to answer whether the remaining throughput gap is in cache
mutation, attention, or the Python loop around the model.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections import defaultdict
from typing import Any

import mlx.core as mx
import mlx_lm


DEFAULT_PROMPT = (
    "Explain the difference between a mutex and a semaphore in practical terms."
)


@dataclasses.dataclass
class Timings:
    total_forward_ms: float = 0.0
    cache_update_ms: float = 0.0
    attention_ms: float = 0.0
    sampler_ms: float = 0.0
    decode_steps: int = 0
    prefill_ms: float = 0.0

    def add(self, other: "Timings") -> None:
        self.total_forward_ms += other.total_forward_ms
        self.cache_update_ms += other.cache_update_ms
        self.attention_ms += other.attention_ms
        self.sampler_ms += other.sampler_ms
        self.decode_steps += other.decode_steps
        self.prefill_ms += other.prefill_ms


def _wrap_method(obj: Any, method_name: str, recorder) -> None:
    original = getattr(obj, method_name)

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        mx.eval(result)
        recorder((time.perf_counter() - t0) * 1000.0)
        return result

    setattr(obj, method_name, wrapped)


def profile_model(
    model_path: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    model, tokenizer = mlx_lm.load(model_path)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    elif hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    else:
        prompt_ids = list(prompt_ids)

    # Monkeypatch the Llama-specific SDPA call so we can time it.
    import mlx_lm.models.base as base_mod
    from mlx_lm.models import cache as cache_mod
    import mlx_lm.models.llama as llama_mod

    timings = Timings()
    phase_timings: dict[str, Timings] = defaultdict(Timings)
    phase = {"name": "prefill"}

    original_sdpa_base = base_mod.scaled_dot_product_attention
    original_sdpa_llama = llama_mod.scaled_dot_product_attention
    decoded: list[int] = []

    def timed_sdpa(*args, **kwargs):
        t0 = time.perf_counter()
        out = original_sdpa_base(*args, **kwargs)
        mx.eval(out)
        delta = (time.perf_counter() - t0) * 1000.0
        timings.attention_ms += delta
        phase_timings[phase["name"]].attention_ms += delta
        return out

    base_mod.scaled_dot_product_attention = timed_sdpa
    llama_mod.scaled_dot_product_attention = timed_sdpa

    try:
        cache = cache_mod.make_prompt_cache(model)
        for layer_cache in cache:
            _wrap_method(
                layer_cache,
                "update_and_fetch",
                lambda delta: (
                    setattr(
                        timings,
                        "cache_update_ms",
                        timings.cache_update_ms + delta,
                    ),
                    setattr(
                        phase_timings[phase["name"]],
                        "cache_update_ms",
                        phase_timings[phase["name"]].cache_update_ms + delta,
                    ),
                ),
            )

        # Prefill
        t0 = time.perf_counter()
        logits = model(mx.array([prompt_ids]), cache=cache)
        mx.eval(logits)
        timings.prefill_ms = (time.perf_counter() - t0) * 1000.0

        # Decode steps
        token_id = mx.argmax(logits[:, -1, :], axis=-1).item()
        decoded = [token_id]
        for _ in range(max_tokens - 1):
            phase["name"] = "decode"
            t0 = time.perf_counter()
            logits = model(mx.array([[token_id]]), cache=cache)
            mx.eval(logits)
            forward_ms = (time.perf_counter() - t0) * 1000.0
            timings.total_forward_ms += forward_ms
            phase_timings["decode"].total_forward_ms += forward_ms
            timings.decode_steps += 1
            phase_timings["decode"].decode_steps += 1

            t1 = time.perf_counter()
            token_id = mx.argmax(logits[:, -1, :], axis=-1).item()
            sampler_ms = (time.perf_counter() - t1) * 1000.0
            timings.sampler_ms += sampler_ms
            phase_timings["decode"].sampler_ms += sampler_ms
            decoded.append(token_id)
            if token_id == getattr(tokenizer, "eos_token_id", None):
                break
    finally:
        base_mod.scaled_dot_product_attention = original_sdpa_base
        llama_mod.scaled_dot_product_attention = original_sdpa_llama

    return {
        "model": model_path,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(decoded),
        "prefill_ms": round(timings.prefill_ms, 2),
        "decode_forward_ms": round(timings.total_forward_ms, 2),
        "cache_update_ms": round(timings.cache_update_ms, 2),
        "attention_ms": round(timings.attention_ms, 2),
        "sampler_ms": round(timings.sampler_ms, 2),
        "decode_steps": timings.decode_steps,
        "per_token": {
            "forward_ms": round(timings.total_forward_ms / max(timings.decode_steps, 1), 3),
            "cache_update_ms": round(timings.cache_update_ms / max(timings.decode_steps, 1), 3),
            "attention_ms": round(timings.attention_ms / max(timings.decode_steps, 1), 3),
            "sampler_ms": round(timings.sampler_ms / max(timings.decode_steps, 1), 3),
        },
        "phase_breakdown": {
            k: {
                "forward_ms": round(v.total_forward_ms, 2),
                "cache_update_ms": round(v.cache_update_ms, 2),
                "attention_ms": round(v.attention_ms, 2),
                "sampler_ms": round(v.sampler_ms, 2),
                "decode_steps": v.decode_steps,
            }
            for k, v in phase_timings.items()
        },
        "decoded": decoded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MLX decode-path cost split")
    parser.add_argument("--model", default="mlx-community/Llama-3.1-8B-Instruct-4bit")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    result = profile_model(args.model, args.prompt, args.max_tokens)
    print("\nDecode profile")
    print(f"model: {result['model']}")
    print(f"prompt_tokens: {result['prompt_tokens']}")
    print(f"generated_tokens: {result['generated_tokens']}")
    print(f"prefill_ms: {result['prefill_ms']}")
    print(f"decode_forward_ms: {result['decode_forward_ms']}")
    print(f"cache_update_ms: {result['cache_update_ms']}")
    print(f"attention_ms: {result['attention_ms']}")
    print(f"sampler_ms: {result['sampler_ms']}")
    print(f"per_token: {result['per_token']}")


if __name__ == "__main__":
    main()
