# mlxz

**High-throughput local inference server for Apple Silicon.**

mlxz serves LLMs on your Mac by adapting vLLM-class serving ideas to MLX on Apple Silicon. The goal is not just to expose an API, but to improve the engine with measurable gains in TTFT, decode throughput, and concurrency over plain `mlx-lm`.

## Why mlxz?

Plain `mlx-lm` processes one request at a time. mlxz adds the serving infrastructure that production workloads need:

| Feature | mlx-lm | mlxz |
|---------|--------|------|
| OpenAI API | No | Yes (`/v1/chat/completions`, streaming SSE) |
| Concurrent requests | No | Yes (continuous batching) |
| Prefix caching | No | Yes (3x TTFT improvement on repeated prompts) |
| Admission control | No | Yes (memory/thermal/queue gating) |
| Metrics & observability | No | Yes (Prometheus, structured logging) |
| Graceful shutdown | No | Yes (drain + signal handlers) |

## Performance

Benchmarked on Apple M3 Max (128 GB) with Llama-3.1-8B-Instruct-4bit:

### Decode Speed
| Metric | mlxz | mlx-lm | Delta |
|--------|------|--------|-------|
| Decode tok/s (32 token gen) | **69.5** | 58.9 | **+18%** |
| Decode tok/s (128 token gen) | **67.8** | 75.1 | -10% |
| Decode tok/s (256 token gen) | **67.4** | 75.1 | -10% |

### Agent Workload (10 requests, shared system prompt)
| Metric | mlxz | mlx-lm | Delta |
|--------|------|--------|-------|
| Avg decode tok/s | **69.5** | 58.9 | **+18%** |
| Total TTFT overhead (10 req) | **1,569 ms** | 5,612 ms | **3.6x faster** |
| TTFT per request (cached) | **151 ms** | 561 ms | **3.7x faster** |
| Concurrent serving | Yes (batch=8) | No | Unique to mlxz |

> On agent workloads with repeated system prompts, mlxz's prefix cache eliminates redundant prefill computation. Combined with better short-generation performance and concurrent serving, that is the path to a genuinely useful MLX engine rather than a thin API wrapper. See [docs/whitepaper.md](docs/whitepaper.md) for the thesis.

## Quick Start

```bash
# Install
git clone https://github.com/metazen11/mlxz && cd mlxz
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras

# Verify environment
uv run mlxz doctor

# Start serving
uv run mlxz serve mlx-community/Llama-3.1-8B-Instruct-4bit

# Apples-to-apples baseline against mlx-lm
uv run mlxz serve mlx-community/Llama-3.1-8B-Instruct-4bit --max-concurrent-requests 1

# Query (OpenAI-compatible)
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b","messages":[{"role":"user","content":"Hello!"}]}'
```

## OpenAI SDK Compatibility

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")

# Non-streaming
response = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Explain quantum computing."}],
    max_tokens=256,
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Write a haiku."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

## Architecture

```
Client (openai-python / curl)
    |
    v
FastAPI + Security Middleware (auth, rate limits, content size)
    |
    v
Admission Controller (memory/thermal/queue gating)
    |
    v
RequestBridge (janus.Queue — async/sync thread boundary)
    |
    v
Engine Thread (SingleStream or ContinuousBatching)
    |-- Prefix Cache (memory + disk tiers, SHA-256 content-addressed)
    |-- Optional Speculative Engine (draft/target)
    |-- Experimental paged-attention modules (not default runtime path)
    |-- Sampling (temperature, top_k, top_p, min_p, greedy)
    v
Token Channel (janus.Queue per request — backpressure)
    |
    v
SSE Stream / JSON Response
```

## Features

### Prefix Caching
Agent workloads (Claude Code, Aider, Cursor) send the same system prompt with every request. mlxz hashes prompt tokens into small blocks by default so short shared prefixes can actually hit cache, then stores the computed KV state. On a cache hit, prefill is skipped entirely — TTFT can drop from hundreds of milliseconds to tens.

### Continuous Batching
Multiple concurrent requests share the GPU. Each engine iteration admits new requests, processes one pending prefill, and batches decode steps.

### Admission Control
Deterministic gate that projects peak KV memory before admitting a request. Rejects with HTTP 429 + resource details when the server would OOM. Also gates on thermal state and queue depth.

### Security
- Bearer auth with `hmac.compare_digest` (constant-time)
- Request body size limits (middleware)
- GGUF file validation before loading
- Security headers on every response
- Secrets never logged (`pydantic.SecretStr`)

### Observability
- Prometheus metrics on separate port (`/metrics` on `:9090`)
- Structured JSON logging via structlog
- Per-request correlation IDs
- Request journal (append-only JSONL)
- Split health probes: `/health/live`, `/health/ready`, `/health/startup`

## CLI

```
mlxz doctor                     # Environment diagnostics
mlxz serve <model> [--port N]   # Start inference server
mlxz bench --regression         # Performance regression check
```

## Benchmarking

```bash
# Start server
uv run mlxz serve mlx-community/Llama-3.1-8B-Instruct-4bit --port 8321 &

# Quick comparison
uv run python benchmarks/compare_to_mlx_lm.py \
  --model mlx-community/Llama-3.1-8B-Instruct-4bit \
  --mlxz-url http://127.0.0.1:8321

# Full matrix benchmark
uv run python benchmarks/run_benchmark.py \
  --model mlx-community/Llama-3.1-8B-Instruct-4bit \
  --mlxz-url http://127.0.0.1:8321 \
  --prompt-tokens 64 256 1024 \
  --max-tokens 32 128 512

# Use `--max-concurrent-requests 1` on the server if you want a pure
# single-stream comparison against mlx-lm.

# Canonical agent-style workload (shared system prompt)
uv run python benchmarks/agent_workload.py \
  --model mlx-community/Llama-3.1-8B-Instruct-4bit \
  --mlxz-url http://127.0.0.1:8321 \
  --requests 10 \
  --max-tokens 128
```

## Testing

```bash
uv run pytest tests/                    # All tests
uv run pytest tests/unit/               # Unit tests only
uv run pytest tests/integration/        # Integration tests
uv run pytest tests/correctness/        # Correctness contract checks
uv run pytest tests/unit/test_block_manager.py  # Hypothesis property tests
```

## Project Structure

```
src/mlxz/
  api/          OpenAI-compatible FastAPI endpoints
  engine/       Inference engines (single-stream, continuous batching)
  scheduler/    Admission control, priority queue, chunked prefill
  prefix_cache/ Content-hashed KV caching (memory + disk tiers)
  paged_attention/ Block manager, paged KV cache
  security/     Auth, input validation, GGUF validator
  observability/ Logging, metrics, request context
  lifecycle/    Graceful shutdown, engine supervision
  profile/      Hardware detection, thermal monitor, residency planner
  telemetry/    SQLAlchemy models, benchmark recorder
  cli/          Typer CLI (doctor, serve, bench)
```

## Roadmap

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 0 | Done | Scaffold, security, observability, CI/CD |
| 1 | Done | Single-stream engine, OpenAI API |
| 2 | Done | Prefix cache (memory + disk) |
| 3 | Experimental | Paged-attention modules and block manager (not default runtime path) |
| 4 | Done | Continuous batching |
| 5 | Done | Speculative decoding engine (runtime-selectable) |
| 6 | Done | Circuit breaker, bench CLI, docs, hardening |

## Thesis

Read [docs/whitepaper.md](docs/whitepaper.md) for the explicit engine thesis: port the useful parts of vLLM's serving model to MLX, measure them honestly, and keep only the ideas that actually improve Apple Silicon inference.

The project contract is in [CONTRACT.md](CONTRACT.md). It defines the continual
test-improve-measure loop that governs performance work until the engine is
consistently better than plain `mlx-lm` on the agreed workloads.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- MLX 0.22.x

## Author

Created by **Mauricio Zuniga** ([@metazen11](https://github.com/metazen11))

## License

Copyright 2026 Mauricio Zuniga. MIT License
