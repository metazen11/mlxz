# Performance Experiments Log

## Baseline Performance (M3 Max 128GB, Llama-3.1-8B-Instruct-4bit)

| Config | mlxz | mlx-lm | Delta |
|--------|------|--------|-------|
| 16 token gen | 69.5 tok/s | 63.2 tok/s | **+10%** |
| 64 token gen | 68.1 tok/s | 72.8 tok/s | -6% |
| 128 token gen | 67.8 tok/s | 74.8 tok/s | -9% |
| 256 token gen | 67.4 tok/s | 75.8 tok/s | -11% |
| Agent workload (10 req, cached) | 69.5 tok/s avg | 58.9 tok/s avg | **+18%** |
| TTFT (prefix cached) | 151 ms | 561 ms | **3.7x faster** |

**Analysis:** mlxz wins on short generations and agent workloads (prefix cache). mlx-lm wins on long isolated generations because it uses `mx.async_eval()` to overlap GPU compute with Python sampling.

---

## Experiment 1: Tight Single-Request Decode Loop

**Branch:** main (merged)
**Date:** 2026-04-18
**Hypothesis:** Reducing Python overhead per iteration by processing 32 tokens in a tight loop instead of 1 token per engine iteration.

**Changes:**
- Single-request fast path decodes 32 tokens before checking for new requests
- Eliminated per-iteration list comprehension and dict iteration

**Result:** 7.7 tok/s -> 24.6 tok/s on 256-token generation (3.2x improvement)
**Status:** MERGED

---

## Experiment 2: Async Token Prefetch (mx.async_eval)

**Branch:** experiment/async-prefetch
**Date:** 2026-04-18
**Hypothesis:** Overlapping GPU compute with Python sampling using mlx-lm's prefetch technique would close the 10% decode gap.

**Changes:**
- `mx.async_eval()` to start next token's forward pass while sampling current
- Dedicated `mx.new_stream()` for generation
- Deferred KV accounting

**Result:** Server crashes on generations > 64 tokens. The KV cache state becomes inconsistent when the model forward pass runs on a dedicated stream while the cache offset is being mutated.

**Root cause:** mlx-lm's async prefetch works because it controls the full lifecycle in a single generator. In mlxz, the cache is shared between the engine thread and the async stream, creating a race on the cache's internal offset.

**Status:** NOT MERGED — needs alternative approach

---

## Experiment 3: mx.compile() for Decode Step (Planned)

**Hypothesis:** Compiling the decode step function with `mx.compile()` should reduce Python dispatch overhead by fusing MLX operations into a single kernel launch.

**Approach:**
```python
@mx.compile
def _compiled_decode_step(model, token_id, cache):
    logits = model(mx.array([[token_id]]), cache=cache)
    return logits
```

**Risk:** `mx.compile()` may not support models with dynamic shapes or KV cache mutations. Need to test.

---

## Experiment 4: Prompt Compilation Cache (Planned)

**Hypothesis:** For repeated prompt patterns (agent workloads), we can cache the compiled computation graph for the prefill step, not just the KV state.

**Approach:** Use `mx.compile()` with shape buckets for common prompt lengths (256, 512, 1024, 2048, 4096, 8192 tokens).

---

## Experiment 5: Tokenizer Batching (Planned)

**Hypothesis:** `tokenizer.decode([token_id])` is called per-token. Batching decode calls (accumulating token IDs and decoding every N tokens) could reduce tokenizer overhead.

**Approach:** Accumulate 8-16 token IDs, batch decode, split into per-token strings for SSE delivery.

---

## Key Insights

1. **Metal compute is NOT the bottleneck.** Both mlxz and mlx-lm use the same MLX Metal kernels. The gap is entirely in Python dispatch overhead.

2. **Async prefetch is fragile with shared cache state.** The KV cache's internal offset is mutated during forward passes, making it unsafe to run concurrent async operations that touch the same cache.

3. **The 32-token tight loop was the biggest win so far.** Reducing per-iteration overhead from the engine loop structure was more impactful than trying to overlap compute.

4. **mlxz's real advantage is architectural, not per-token.** Prefix caching, concurrent serving, and admission control provide value that raw tok/s doesn't capture.
