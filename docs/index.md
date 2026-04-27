# mlxz

**High-throughput local inference server for Apple Silicon.**

mlxz serves LLMs on your Mac by porting useful vLLM-class serving ideas to MLX and keeping only the ones that move benchmarks.

## Features

- **OpenAI API** — Drop-in replacement for `openai-python` SDK
- **Prefix Caching** — 3x TTFT improvement on repeated prompts
- **Continuous Batching** — Serve multiple concurrent requests
- **Speculative Decoding** — Runtime-selectable draft/target engine
- **Paged Attention Modules** — Experimental block-manager + paged KV components
- **Admission Control** — Prevents OOM under load
- **Production Observability** — Prometheus metrics, structured logging

## Thesis

The project is engine-first. The public API and observability are support systems for the real goal: better TTFT, decode throughput, concurrency, and correctness on Apple Silicon. Start with [whitepaper.md](whitepaper.md) for the explicit thesis and [contract.md](contract.md) for the operating loop.

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
