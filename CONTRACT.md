# mlxz Project Contract

This repository exists to improve the `mlxz` engine until it demonstrates a
clear, repeatable advantage over plain `mlx-lm` on the workloads that matter.

The goal is not to ship features for their own sake. The goal is to make the
engine measurably better, then keep iterating until that advantage is strong
enough to support real adoption.

## Core Objective

- Improve the `mlxz` engine, not just the API.
- Beat plain `mlx-lm` on the agreed benchmark matrix whenever the change is
  intended to improve performance.
- Keep the project honest about where the win exists: single-stream, repeated
  prefix, concurrent agent workloads, memory fit, or correctness.

## Operating Loop

Every meaningful engine change must go through this cycle:

1. Identify the bottleneck with profiling or benchmark evidence.
2. Form a concrete hypothesis about the smallest useful change.
3. Implement the change behind the least risky path that can prove it.
4. Run correctness checks before performance claims.
5. Benchmark against `mlx-lm` on the project matrix.
6. Record the result, including failures and rejected ideas.
7. Keep the change only if it moves the target metric without unacceptable
   regression.

## Required Benchmarks

Performance work should be measured on:

- `Llama-3.2-3B-Instruct-4bit`
- `Llama-3.1-8B-Instruct-4bit`
- `Qwen2.5-14B-Instruct-4bit`

And on the workloads that reflect the product thesis:

- single-request decode
- repeated-prefix agent traffic
- moderate concurrency / continuous batching

If a change only helps one model or one workload, say so plainly. Do not
generalize it beyond the evidence.

## Merge Gate

Do not merge performance-related changes unless all of the following are true:

- correctness tests pass
- benchmark results are recorded
- the result is compared against `mlx-lm`
- the docs or experiment log are updated
- any rejected approach is documented

## Improvement Rule

This project is always in improvement mode until the engine is consistently
better than plain MLX on the workloads we care about. If a new idea does not
move the engine, it belongs in the experiments log or the docs, not in the
default path.

## Exit Criteria For A Given Sprint

A sprint is only done when:

- the change is benchmarked
- the benchmark result is written down
- the relevant docs are updated
- the next bottleneck is identified

