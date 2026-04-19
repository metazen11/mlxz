# Architecture

## Overview

mlxz is a serving engine, not a training framework. It loads models via mlx-lm and serves them over an OpenAI-compatible HTTP API.

## Components

### Engine Layer
- **SingleStreamEngine** — Batch=1, synchronous. Simple and fast for single-user.
- **ContinuousBatchingEngine** — Iteration-level batching for concurrent requests.
- **SpeculativeEngine** — Runtime-selectable draft-target engine with rejection sampling.

### Cache Layer
- **PrefixCacheMemory** — In-memory LRU with content-addressed hashing
- **PrefixCacheDisk** — Persistent cache with safetensors + SHA-256 checksums
- **BlockManager / PagedKVCache** — Experimental paged-attention primitives

### API Layer
- **FastAPI** — OpenAI-compatible endpoints with SSE streaming
- **Admission Controller** — Memory/thermal/queue gating
- **Security Middleware** — Bearer auth, content limits, GGUF validation

### Observability
- **Prometheus** — Metrics on separate port
- **structlog** — JSON structured logging with secret redaction
- **Request Journal** — Append-only JSONL for post-mortem
