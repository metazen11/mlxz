# Performance Experiments Log

This log exists to keep the project honest about the original thesis: port serving ideas from vLLM to MLX, measure them on real workloads, and keep the ones that actually improve the engine.

## Baseline Performance (M3 Max 128GB, Llama-3.1-8B-Instruct-4bit)

Bench harness notes:
- Benchmarks run with `temperature=0` and fixed seed for deterministic decode.
- `mlx-lm` TTFT is measured via `mlx_lm.stream_generate()` first-token timing.

| Config | mlxz | mlx-lm | Delta |
|--------|------|--------|-------|
| 16 token gen | 69.5 tok/s | 63.2 tok/s | **+10%** |
| 64 token gen | 68.1 tok/s | 72.8 tok/s | -6% |
| 128 token gen | 67.8 tok/s | 74.8 tok/s | -9% |
| 256 token gen | 67.4 tok/s | 75.8 tok/s | -11% |
| Agent workload (10 req, cached) | 69.5 tok/s avg | 58.9 tok/s avg | **+18%** |
| TTFT (prefix cached) | 151 ms | 561 ms | **3.7x faster** |

**Analysis:** mlxz wins on short generations and agent workloads (prefix cache). mlx-lm wins on long isolated generations because it uses `mx.async_eval()` to overlap GPU compute with Python sampling.

**Comparison rule:** For apples-to-apples engine comparisons, run mlxz in single-stream mode with `--max-concurrent-requests 1`. Measure continuous batching separately, because concurrency is a product feature, not part of the pure single-request baseline.

**Current concurrent-serving checkpoint:** On an 8-request agent-style workload in continuous mode, the server held a median TTFT of 820.5 ms and aggregate throughput of 64.62 tok/s. That is not a raw decode win, but it is a real multi-request serving result that `mlx-lm` does not provide.

**Current repeated-prefix checkpoint:** After storing chunk-boundary prefix states instead of only full prompts, the sequential agent-style workload on the 8B model improved to a median TTFT of 71.4 ms, compared with 104.1 ms for `mlx-lm`. Decode throughput is still slightly behind (`71.86 tok/s` vs `76.97 tok/s`), so the engine is not a blanket throughput win yet, but the prefix-cache story is now real instead of theoretical.

## Open Experiment Issues

The ideas we still want to try are tracked in GitHub so they do not get lost:

- [#11 MLX cache mutation hot path](https://github.com/metazen11/mlxz/issues/11)
- [#12 true multi-request packed forward pass](https://github.com/metazen11/mlxz/issues/12)
- [#13 custom Metal attention kernel for paged blocks](https://github.com/metazen11/mlxz/issues/13)
- [#14 smarter prefix cache reuse for sub-256-token prompts](https://github.com/metazen11/mlxz/issues/14)
- [#15 quantized KV cache start for long prompts](https://github.com/metazen11/mlxz/issues/15)

## What Still Needs To Happen

- Compile or shape-bucket the stable hot paths only if they survive a multi-model test matrix.
- Keep testing on at least two additional models, not just the 8B headline model.
- Measure the workloads that matter to the thesis: short generation, long generation, repeated-prefix agent traffic, and moderate concurrency.
- Only keep an optimization if it survives both correctness checks and multi-model benchmarks.

## Rejected / Archived Ideas

When an optimization fails, it belongs here. Keep the record short and factual:

- What was tried
- Which model(s) and workload(s) were used
- What broke or failed to improve
- Why we rejected it

Current entry:

- `mx.async_eval()` prefetch in the shared engine/cache path: crashed on generations over 64 tokens because cache state became inconsistent across streams.
- `mx.compile()` on the decode step: safe, but only a small win. On the current matrix it improved 8B by about 1.6%, 3B by about 0.3%, and 14B by about 1.4%. Not enough to close the mlx-lm gap by itself.
- Speculative decoding with a 3B draft model on the 8B target: started after wiring `set_prefix_cache()` into the engine, but the current implementation was slower than plain single-stream decoding and worse than `mlx-lm` on the same benchmark. Keep it out of the default path until the draft-target cache reuse story is stronger.
- KV cache growth-step sweep (`256` vs `1024` on the 8B long-prompt server benchmark): no material win. `mlxz` stayed behind `mlx-lm` on the 1024-token prompt/128-token generation run in both configurations, so the knob is not worth carrying forward as a default optimization.

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

**Measured result:** The compiled decode step is valid, but the end-to-end gain was small:
- 8B: ~1.6% faster than the plain decode loop
- 3B: ~0.3% faster
- 14B: ~1.4% faster

**Status:** NOT ENOUGH TO SHIP as a primary optimization.

---

## Experiment 4: Prompt Compilation Cache (Planned)

**Hypothesis:** For repeated prompt patterns (agent workloads), we can cache the compiled computation graph for the prefill step, not just the KV state.

**Approach:** Use `mx.compile()` with shape buckets for common prompt lengths (256, 512, 1024, 2048, 4096, 8192 tokens).

---

## Experiment 5: Tokenizer Batching (Planned)

**Hypothesis:** `tokenizer.decode([token_id])` is called per-token. Batching decode calls (accumulating token IDs and decoding every N tokens) could reduce tokenizer overhead.

**Approach:** Accumulate 8-16 token IDs, batch decode, split into per-token strings for SSE delivery.

## Experiment 6: Multi-Model Benchmark Matrix

**Hypothesis:** An optimization that only helps one model size is probably not a general engine improvement. The right bar is stable or improved results on at least one smaller model and one larger model, compared with plain `mlx-lm`.

**Approach:** Run the same benchmark matrix on `Llama-3.2-3B-Instruct-4bit`, `Llama-3.1-8B-Instruct-4bit`, and `Qwen2.5-14B-Instruct-4bit` with deterministic sampling and the canonical agent workload.

**Status:** Planned. This should become part of the merge gate for performance work.

## Experiment 7: Speculative Decoding (Draft-Target)

**Hypothesis:** A smaller draft model should let the target model verify multiple candidate tokens per pass and reduce end-to-end latency.

**Setup:** 8B target model + 3B draft model, `num_draft_tokens=4`.

**Measured result:** The current implementation did not outperform the single-stream baseline. On the 8B benchmark prompt, speculative mode was slower than plain `mlxz` and also slower than `mlx-lm`.

**Status:** NOT A DEFAULT OPTIMIZATION. The engine now accepts the shared prefix-cache wiring used at startup, but the draft-target cache reuse path still needs real work before this becomes competitive.

---

## Experiment 8: Greedy Decode Chunking + Continuous Starvation Fix

**Hypothesis:** The remaining single-request decode gap is mostly Python boundary overhead, and the concurrent agent workload should not be allowed to enter the single-request fast path while other requests are already resident.

**Changes:**
- Compiled greedy decode now emits fixed-size chunks instead of one token per call in the single-request path.
- Continuous mode only uses the single-request fast path when it is actually serving one request total.

**Measured result on `Llama-3.1-8B-Instruct-4bit`:**
- Single-request benchmark: `mlxz` 76.0 tok/s vs `mlx-lm` 77.0 tok/s, with TTFT dropping to 38.1 ms from 137.8 ms.
- Agent workload, 8 concurrent requests: median TTFT 541.0 ms, p95 TTFT 879.7 ms, median decode throughput 8.90 tok/s, aggregate throughput 68.35 tok/s.
- Removing redundant state evals around compiled decode calls did not materially change throughput; the run stayed at single-request parity and the concurrent workload remained in the same band.
- Increasing SSE delivery batching and trimming hot-path attribute lookups did not materially move throughput either; the agent workload stayed in the same 67-68 tok/s band.

**Status:** SHIPPED FOR NOW. The decode sweep improved latency materially and preserved throughput parity on the isolated path, but it did not produce a clear aggregate-throughput win over the concurrent baseline.

---

## Experiment 9: Greedy Chunk Size Sweep

**Hypothesis:** Increasing the fixed greedy decode chunk size would amortize more Python/model-call overhead in the single-request path without affecting the concurrent path.

**Change:** Increased the compiled greedy chunk from 8 to 16 tokens in the single-request decode fast path.

**Measured result:** No material throughput gain.
- Single-request benchmark remained in the same band as the prior run: `mlxz` 76.4 tok/s vs `mlx-lm` 77.1 tok/s.
- Agent workload, 8 concurrent requests: aggregate throughput 67.57 tok/s, median TTFT 599.9 ms.

**Status:** NOT A MATERIAL WIN. Keep the larger chunk size only if later runs show it is still neutral; otherwise revert it during the next cleanup pass.

---

## Experiment 10: Cache-Class Swap (`KVCache` vs `RotatingKVCache`)

**Hypothesis:** Using `RotatingKVCache` instead of the default append-style cache might reduce decode-side cache overhead enough to matter on larger models.

**Measured result:** No meaningful decode improvement.
- 8B: `KVCache` 73.08 tok/s vs `RotatingKVCache256` 73.15 tok/s.
- 3B: `KVCache` 140.62 tok/s vs `RotatingKVCache256` 141.64 tok/s.
- 14B: `KVCache` 39.59 tok/s vs `RotatingKVCache256` 39.53 tok/s.

**Status:** REJECTED. The cache implementation swap does not solve the decode bottleneck; the main work still needs to happen in MLX primitives or a true multi-request batching strategy.

---

## Experiment 11: Exact-Offset Batched Decode

**Hypothesis:** Requests that are already at the same cache offset can share one forward pass, letting the engine amortize cache update and attention work across compatible requests without changing sampling semantics.

**Change:** Group decode requests by current cache offset, batch their caches into one MLX call, then scatter the updated cache state back to each request.

**Measured result on `Llama-3.1-8B-Instruct-4bit`:**
- Single-request benchmark stayed in the same band: `mlxz` 76.5 tok/s vs `mlx-lm` 77.2 tok/s.
- Agent workload, 8 concurrent requests, improved to median TTFT 597.1 ms, p95 TTFT 1001.9 ms, median decode throughput 11.00 tok/s, aggregate throughput 83.28 tok/s, wall time 12.30 s.

**Status:** PROMISING. This is the first Python-level batching change that materially improved the concurrent agent workload. The remaining question is how much more benefit we can get by widening the compatibility rule or moving the same idea into MLX itself.

---

## Experiment 12: Quantized KV Cache Start

**Hypothesis:** Starting long-context requests on `QuantizedKVCache` once the prompt crosses `quantized_kv_start` would reduce cache-bandwidth pressure on the decode hot path without changing sampling semantics.

**Change:** Wired `RuntimeConfig.kv.quantized_kv_start` into cache construction so long prompts start on the quantized cache path from the beginning, and tagged prefix-cache entries by cache type so short and long cache states do not mix.

**Measured result on 1024-token prompts with `max_tokens=128`:**
- `Llama-3.1-8B-Instruct-4bit`: `mlxz` 65.8 tok/s vs `mlx-lm` 37.8 tok/s, with total latency 4104 ms vs 4894 ms.
- `Llama-3.2-3B-Instruct-4bit`: `mlxz` 64.0 tok/s vs `mlx-lm` 76.8 tok/s, with total latency 3218 ms vs 2807 ms.

**Implementation note:** The local `QuantizedKVCache` wrapper now restores `offset` when prefix-cache state is assigned. Upstream `mlx_lm` preserves the quantized buffers but leaves `offset` stale, which would make restored caches inconsistent after a prefix hit.

**Follow-up benchmark on the current branch (`Llama-3.1-8B-Instruct-4bit`, prompt~1228, max_tokens=128, 3 runs):**
- `mlxz`: `62.0 tok/s`, TTFT `129.4 ms`, total `2195.1 ms`
- `mlx-lm`: `32.2 tok/s`, total `7027.0 ms`

**Additional follow-up on `Qwen2.5-14B-Instruct-4bit`, prompt~1222, max_tokens=128, 3 runs:**
- `mlxz`: `18.4 tok/s`, TTFT `590.6 ms`, total `7550.4 ms`
- `mlx-lm`: `17.5 tok/s`, total `11160.2 ms`

**Counterexample on `Llama-3.2-3B-Instruct-4bit`, prompt~1228, max_tokens=128, 3 runs:**
- `mlxz`: `54.8 tok/s`, TTFT `88.1 ms`, total `2424.7 ms`
- `mlx-lm`: `67.9 tok/s`, total `2899.6 ms`

**Status:** PARTIAL WIN. This is a real win on the longer 8B and 14B contexts, but it does not generalize to the 3B run. Keep the wiring, keep measuring 3B/14B, and treat this as a workload-dependent optimization rather than a universal throughput win.

---

## Experiment 13: Adaptive Quantized KV Policy Sweep

**Hypothesis:** A fixed quantized-cache start point is too coarse. Long prompts want quantization from the start, while short prompts are harmed by it, so the policy should likely depend on request length or total requested sequence length.

**Change:** Use total requested length (`prompt_token_count + max_tokens`) to decide whether to start on `QuantizedKVCache`, instead of looking only at prompt length.

**Measured result on long prompts (`prompt~1024`, `max_tokens=128`):**
- `Llama-3.1-8B-Instruct-4bit`: `kv_start=0` was better than `256` (`70.0 tok/s` vs `59.8 tok/s`).
- `Llama-3.2-3B-Instruct-4bit`: `kv_start=0` was better than `256` (`125.0 tok/s` vs `114.7 tok/s`).
- `Qwen2.5-14B-Instruct-4bit`: `kv_start=0` was better than `256` (`37.3 tok/s` vs `32.4 tok/s`), but still slower than `mlx-lm` on that model (`22.8 tok/s` vs `36.1 tok/s` when compared directly).

**Measured result on a short prompt (`prompt~64`, `max_tokens=128`, 8B):**
- `kv_start=0` improved over `kv_start=256` on the same harness (`37.6 tok/s` vs `27.7 tok/s`), but the short-prompt path still lost to `mlx-lm` (`59.1 tok/s`).

**Status:** IMPLEMENTED AS A BETTER DEFAULT, BUT STILL NOT A UNIVERSAL WIN. The engine now uses total requested length to avoid the worst short-prompt penalty, while keeping the long-context gains. The remaining question is whether this should stay as a fixed threshold, become model-specific, or evolve into a more dynamic policy.

---

## Key Insights

1. **Metal compute is NOT the bottleneck.** Both mlxz and mlx-lm use the same MLX Metal kernels. The gap is entirely in Python dispatch overhead.

2. **Async prefetch is fragile with shared cache state.** The KV cache's internal offset is mutated during forward passes, making it unsafe to run concurrent async operations that touch the same cache.

3. **The 32-token tight loop was the biggest win so far.** Reducing per-iteration overhead from the engine loop structure was more impactful than trying to overlap compute.

4. **mlxz's real advantage is architectural, not per-token.** Prefix caching, concurrent serving, and admission control provide value that raw tok/s doesn't capture.
