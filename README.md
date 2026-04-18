# mlxz

**High-throughput local inference server for Apple Silicon.**

mlxz serves LLMs on your Mac with vLLM-class features — paged attention, continuous batching, prefix caching — over an OpenAI-compatible API. No CUDA, no cloud, no training. Just fast local inference.

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

| Metric | mlxz | mlx-lm | Delta |
|--------|------|--------|-------|
| Decode tok/s (short gen) | **71.7** | 61.9 | +16% |
| Decode tok/s (long gen) | **69.4** | 61.6 | +13% |
| TTFT (cold) | 183 ms | 14 ms | Higher (HTTP overhead) |
| TTFT (prefix cache hit) | **61 ms** | N/A | 3x faster than cold |
| Concurrent throughput | **batch=4** | batch=1 only | 3-4x aggregate |

> Decode speedup comes from our tighter decode loop with less Python overhead per token. TTFT is higher on cold requests due to the HTTP stack, but prefix caching eliminates this for agent workloads with repeated system prompts.

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
    |-- Block Manager (paged attention, refcounted, COW)
    |-- Sampling (temperature, top_k, top_p, min_p, greedy)
    v
Token Channel (janus.Queue per request — backpressure)
    |
    v
SSE Stream / JSON Response
```

## Features

### Prefix Caching
Agent workloads (Claude Code, Aider, Cursor) send the same system prompt with every request. mlxz hashes prompt tokens into 256-token blocks and caches the computed KV state. On a cache hit, prefill is skipped entirely — TTFT drops from hundreds of milliseconds to tens.

### Continuous Batching
Multiple concurrent requests share the GPU. Each engine iteration admits new requests, processes prefill chunks, and batches decode steps. Chunked prefill prevents head-of-line blocking.

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
```

## Testing

```bash
uv run pytest tests/                    # All tests (254 passing)
uv run pytest tests/unit/               # Unit tests only
uv run pytest tests/integration/        # Integration tests
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
| 3 | Done | Paged attention, block manager |
| 4 | Done | Continuous batching |
| 5 | Planned | Speculative decoding |
| 6 | Planned | Hardening, Homebrew, soak tests |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- MLX 0.22.x

## License

Apache-2.0
