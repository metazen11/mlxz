# mlxz

**High-throughput local inference server for Apple Silicon.**

mlxz serves LLMs on your Mac with vLLM-class features over an OpenAI-compatible API.

## Features

- **OpenAI API** — Drop-in replacement for `openai-python` SDK
- **Prefix Caching** — 3x TTFT improvement on repeated prompts
- **Continuous Batching** — Serve multiple concurrent requests
- **Speculative Decoding** — 1.5-2.5x effective throughput with draft model
- **Paged Attention** — Memory-efficient KV cache with reference counting
- **Admission Control** — Prevents OOM under load
- **Production Observability** — Prometheus metrics, structured logging

## Quick Start

```bash
pip install mlxz  # or: uv pip install mlxz
mlxz serve mlx-community/Llama-3.1-8B-Instruct-4bit
```

Then query with any OpenAI-compatible client:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -d '{"model":"llama","messages":[{"role":"user","content":"Hello!"}]}'
```

## Architecture

See [Architecture](architecture.md) for the full design.
