# mlxz Benchmarks

Performance benchmarking suite for comparing mlxz inference against plain mlx-lm.

## Hardware Requirements

- Apple Silicon Mac (M1/M2/M3/M4 family)
- Minimum 16 GB unified memory (32 GB+ recommended for 8B models)
- macOS 14+ with Metal support
- Sufficient disk space for the model (~5 GB for a 4-bit 8B model)

## Quick Start

```bash
# 1. Start the mlxz server
mlxz serve mlx-community/Llama-3.1-8B-Instruct-4bit --port 8321

# 2. In another terminal, run the quick comparison
python benchmarks/compare_to_mlx_lm.py \
    --model mlx-community/Llama-3.1-8B-Instruct-4bit

# 3. Or run the full benchmark matrix
python benchmarks/run_benchmark.py \
    --model mlx-community/Llama-3.1-8B-Instruct-4bit \
    --mlxz-url http://127.0.0.1:8321
```

## Scripts

### `run_benchmark.py` -- Full Matrix Benchmark

Runs a configurable matrix of (prompt_tokens x max_tokens) against both mlxz
and mlx-lm, taking the median of N runs per configuration.

**Metrics collected:**
- **TTFT (ms):** Time-to-first-token -- measured from request send to first
  SSE content chunk for mlxz; approximated for mlx-lm.
- **Decode (tok/s):** Tokens per second during the decode phase.
- **Total latency (ms):** Wall-clock time from request to last token.
- **Completion tokens:** Actual number of tokens generated.

**Key flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `mlx-community/Llama-3.1-8B-Instruct-4bit` | HuggingFace model ID |
| `--mlxz-url` | `http://127.0.0.1:8321` | mlxz server address |
| `--prompt-tokens` | `64 256 1024 4096` | Prompt sizes to test |
| `--max-tokens` | `32 128 512` | Max completion lengths |
| `--runs` | `3` | Runs per config (median kept) |
| `--skip-mlx-lm` | off | Skip mlx-lm comparison |
| `--skip-mlxz` | off | Skip mlxz (mlx-lm only) |

### `compare_to_mlx_lm.py` -- Quick A/B Comparison

Runs a single prompt through both systems and prints a side-by-side table.
Ideal for quick sanity checks during development.

## Results Directory

Benchmark results are saved as timestamped JSON files in `benchmarks/results/`:

```
benchmarks/results/benchmark_20260417_143022.json
```

Each file contains an array of `BenchmarkResult` objects:

```json
[
  {
    "system": "mlxz",
    "model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "prompt_tokens": 64,
    "max_tokens": 128,
    "completion_tokens": 128,
    "ttft_ms": 45.2,
    "decode_tps": 78.3,
    "total_latency_ms": 1680.5,
    "timestamp": "2026-04-17T14:30:22"
  }
]
```

## Baseline and Regression Detection

The first time `run_benchmark.py` executes, it saves results as
`benchmarks/baseline.json`.  On subsequent runs, current results are compared
against this baseline and any metric that drops more than 10% is flagged as a
regression:

```
--- Regression Check vs Baseline ---
  mlxz prompt=64 max=128: 78.3 vs 80.1 tok/s (0.98x) [OK]
  mlxz prompt=1024 max=128: 55.2 vs 72.0 tok/s (0.77x) [REGRESSION]
```

To reset the baseline after an intentional change:

```bash
rm benchmarks/baseline.json
python benchmarks/run_benchmark.py --model <your-model>
```

## Interpreting Results

### What matters most

1. **Decode tok/s** is the primary throughput metric.  Higher is better.
   mlxz targets parity or improvement over plain mlx-lm through KV-cache
   quantisation, prefix caching, and continuous batching.

2. **TTFT** matters for interactive use.  mlxz adds HTTP overhead but may
   offset it with prefix cache hits on repeated prompts.

3. **Total latency** is the user-perceived wall-clock time.

### Expected overhead

mlxz adds a thin HTTP + SSE layer on top of the raw MLX compute path.
For small `max_tokens` values (< 32), HTTP framing overhead may dominate.
For longer generations (128+ tokens), decode throughput should converge
with or exceed plain mlx-lm due to engine optimizations.

### Thermal throttling

Apple Silicon throttles under sustained load.  If you see inconsistent
results, check `powermetrics` or the `thermal_state` field in mlxz
telemetry.  The benchmark takes the median of N runs to mitigate this.

## CI Integration

Add to your CI pipeline (self-hosted Apple Silicon runner):

```yaml
- name: Performance regression check
  run: |
    mlxz serve mlx-community/Llama-3.1-8B-Instruct-4bit --port 8321 &
    sleep 10
    python benchmarks/run_benchmark.py \
      --model mlx-community/Llama-3.1-8B-Instruct-4bit \
      --prompt-tokens 64 256 \
      --max-tokens 32 128 \
      --runs 3
```
