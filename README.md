# mlxz

High-throughput local inference server for Apple Silicon. Engine-only, no training.

Ships vLLM-class serving semantics (paged attention, continuous batching, prefix caching, speculative decoding) on MLX with an OpenAI-compatible API.

## Quick Start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone ... && cd mlxz
uv sync --all-extras
uv run mlxz doctor
uv run mlxz serve --model mlx-community/Llama-3.1-8B-Instruct-4bit
```

## Status

Under active development. See `MLXZ_IMPLEMENTATION_PLAN.md` for the full roadmap.

## License

Apache-2.0
