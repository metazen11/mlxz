# `mlxz` — Full Git Repository Implementation Plan

> High-throughput local inference server for Apple Silicon. Engine-only, no training required. Ships vLLM-class serving semantics (paged attention, continuous batching, prefix caching, OpenAI API) on MLX.
>
> This document is the single source of truth for repo structure, tech stack, module specs, TDD harness, CI/CD, benchmarking, branching, and phased milestones. It pairs with `docs/whitepaper.md` (the "why") as the "how."

---

## 0. Project philosophy and non-goals

**Philosophy.** Small, Python-first, research-velocity codebase. Every new module ships with tests and a benchmark before it lands on `main`. No module is merged unless it demonstrably moves a benchmark number or closes a correctness gap. Correctness regressions (PPL, HumanEval, KL-vs-baseline, prefix-cache determinism) fail CI unconditionally. Performance regressions >5% on the reference workload fail CI on the self-hosted runner.

**Ordering principle: ship user value every phase.** Phase 2 (prefix caching on contiguous KV) lands before Phase 3 (paged attention) because prefix caching delivers the biggest single UX win and doesn't require paged attention to function. This is an intentional departure from how vLLM was built bottom-up.

**Explicit non-goals.**
- Multi-GPU / distributed inference. (Single-node Apple Silicon is the target.)
- Training or fine-tuning. **Zero learned components in v1.0.**
- CUDA / ROCm / CPU-only backends.
- Swift / iOS / visionOS ports. (Deferred to v2.)
- MoE architectures. (Dense-only for v1; MoE is v2.)
- Embedding endpoint (`/v1/embeddings`). (v1.1.)
- Multi-model loading in a single process. (v2; 64 GB memory math basically forces one-at-a-time.)
- Custom tool-calling server semantics beyond pass-through of OpenAI's message format.

**Audience.** ML systems engineers comfortable with MLX, `uv`, pytest, FastAPI, and Metal instrumentation. Not an end-user product.

---

## 1. Tech stack and justifications

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Matches MLX / mlx-lm; PEP 695, `tomllib`, faster interpreter |
| Runtime | MLX (pinned) | UMA semantics, `QuantizedKVCache`, lazy eval |
| API layer | FastAPI + uvicorn | Async, Pydantic-native, standard, large ecosystem |
| Package / env mgr | `uv` | Fast, lockfile-first, single-binary |
| Packaging | `hatchling` via `pyproject.toml` | PEP 621 native |
| Testing | `pytest` + `pytest-xdist` + `hypothesis` + `pytest-benchmark` | Property tests for block manager/scheduler/prefix cache state machines |
| Type checking | `pyright` (strict on `src/`) | Faster than mypy; better inference |
| Lint / format | `ruff` (lint + format) | Single tool |
| Telemetry DB | SQLAlchemy 2.x + Alembic (SQLite default, Postgres optional) | Matches your Django/Postgres stack; fits WFCA infra if desired |
| Config | `pydantic-settings` v2 | TOML + env overrides, validated |
| CLI | `typer` | Auto-help, type-driven |
| Logging | `structlog` → JSON → stdout | Grep-able, pipe to Postgres or file |
| Metrics | `prometheus-client` | Standard `/metrics` endpoint |
| Load testing | `locust` | Agent-workload replay |
| GGUF interop | `gguf` (ggml-org's Python reader) | Canonical parser |
| CI | GitHub Actions + self-hosted M-series runner | Reference-hardware gating |
| Docs | `mkdocs-material` + `mkdocs-mermaid2` | Good defaults, deploy to GH Pages |
| Release | `release-please` (Conventional Commits) | Automated CHANGELOG + semver |

**Rationale for Python-only v1 (no C++/Swift).** MLX's Python API exposes everything needed. Dropping to C++ buys ~2–5% on hot-path dispatch overhead but costs 10× iteration velocity. When profiling shows dispatch overhead is actually material, rewrite only the identified hot loop as a custom `mx.fast` primitive — don't port the whole runtime.

**Rationale for FastAPI over `aiohttp` or raw Starlette.** Pydantic integration is the decisive factor. Request/response schemas are the public contract; type-checked Pydantic models are the cheapest way to maintain that contract. OpenAI-API compatibility has a lot of optional fields and FastAPI's schema generation catches mistakes before clients do.

**Rationale for SQLite default, Postgres optional.** Phase 0 users should be able to clone, `uv sync`, and `mlxz bench` without setting up a database. The schema is written in portable SQLAlchemy so promoting to Postgres (your WFCA infra) is a config-only change.

---

## 2. Repository layout

```
mlxz/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # lint, typecheck, unit, tiny-integration (hosted)
│   │   ├── bench.yml                 # full benchmark gate (self-hosted M4 Max)
│   │   ├── correctness.yml           # weekly MMLU/HumanEval/NIAH + soak test
│   │   └── release-please.yml        # auto-versioning
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   ├── perf_regression.yml       # structured: model, quant, hw, tok/s delta
│   │   └── feature_request.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── CODEOWNERS
├── src/mlxz/
│   ├── __init__.py                   # public API re-exports; __version__
│   ├── config.py                     # Pydantic Settings + TOML loader
│   ├── types.py                      # shared dataclasses, Enums, Protocols
│   ├── exceptions.py                 # AdmissionRejected, ResidencyOverflow, DraftIncompatible, etc.
│   │
│   ├── api/                          # FastAPI app and OpenAI-compat endpoints
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPI app factory; DI wiring
│   │   ├── openai.py                 # /v1/chat/completions, /v1/completions, /v1/models
│   │   ├── health.py                 # /health/{live,ready,startup} — split probes
│   │   ├── metrics.py                # /metrics (Prometheus registry)
│   │   └── schemas.py                # Pydantic request/response models (OpenAI-compatible)
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── request.py                # Request dataclass + lifecycle state machine
│   │   ├── single_stream.py          # Phase 1 (batch=1, sync)
│   │   ├── continuous.py             # Phase 4 (iteration-level batching)
│   │   ├── speculative.py            # Phase 5 (draft-target + rejection sampling)
│   │   └── sampling.py               # temperature, top_p, top_k, min_p
│   │
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── admission.py              # projects peak memory; accepts or rejects
│   │   ├── priority.py               # FCFS with optional priority classes
│   │   └── chunker.py                # chunked-prefill splits (Sarathi-Serve style)
│   │
│   ├── paged_attention/              # Phase 3
│   │   ├── __init__.py
│   │   ├── block_manager.py          # block pool, free list, refcount, COW
│   │   ├── paged_kv.py               # block-table-backed KV cache
│   │   └── attention.py              # gather from blocks → mx.fast.SDPA
│   │
│   ├── prefix_cache/
│   │   ├── __init__.py
│   │   ├── base.py                   # PrefixCacheProtocol
│   │   ├── hasher.py                 # rolling SHA-256 at block boundaries
│   │   ├── memory.py                 # Phase 2 memory tier (contiguous KV slices)
│   │   ├── disk.py                   # Phase 2 disk tier (16KB-aligned, mmap)
│   │   └── block_backed.py           # Phase 3 upgrade: shares physical blocks
│   │
│   ├── cache/                        # non-prefix KV cache policies
│   │   ├── __init__.py
│   │   ├── base.py                   # KVCacheProtocol
│   │   ├── quantized.py              # wraps mlx-lm QuantizedKVCache
│   │   ├── streaming.py              # attention-sink preservation (first-N pin)
│   │   └── kivi.py                   # per-channel K / per-token V INT4 (opt-in)
│   │
│   ├── loader/
│   │   ├── __init__.py
│   │   ├── safetensors_store.py      # aligned loader
│   │   ├── gguf_bridge.py            # GGUF → MLX with K-quant-mimicking recipe
│   │   └── quant_recipe.py           # mixed-precision recipe parser
│   │
│   ├── profile/
│   │   ├── __init__.py
│   │   ├── hardware.py               # chip detection, mx.metal.device_info
│   │   ├── residency.py              # wired-limit probe, budget derivation
│   │   ├── thermal.py                # powermetrics sampling
│   │   └── profiler.py               # per-layer timing, BW probe
│   │
│   ├── metal/
│   │   ├── __init__.py
│   │   ├── compile_cache.py          # mx.compile per-shape bucket table
│   │   └── flash_decoding.py         # split-KV-along-seq attention for long ctx (Phase 6 stretch)
│   │
│   ├── telemetry/
│   │   ├── __init__.py
│   │   ├── db.py                     # SQLAlchemy engine + session
│   │   ├── models.py                 # Run, Request, Measurement tables
│   │   ├── recorder.py               # context manager for benchmark runs
│   │   └── alembic/                  # migrations
│   │
│   ├── lifecycle/
│   │   ├── __init__.py
│   │   ├── shutdown.py               # ShutdownCoordinator, drain, SIGTERM
│   │   └── supervisor.py             # EngineThreadSupervisor, crash recovery
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   ├── limits.py                 # RequestLimits (max tokens, body size)
│   │   ├── auth.py                   # BearerAuthMiddleware (hmac.compare_digest)
│   │   ├── gguf_validator.py         # Pre-parse GGUF size/shape validation
│   │   └── headers.py                # Security headers middleware
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── context.py                # RequestContext, correlation IDs
│   │   ├── logging.py                # structlog config, secret redaction
│   │   └── journal.py                # Request journal (append-only JSONL)
│   │
│   └── cli/
│       ├── __init__.py
│       ├── main.py                   # `mlxz` entry point
│       ├── doctor.py                 # environment diagnostics (--smoke for full-stack test)
│       ├── serve.py                  # start the server
│       ├── bench.py                  # matrix bench + regression mode
│       ├── convert.py                # apply quant recipe to a checkpoint
│       ├── replay.py                 # replay a captured agent trace for load testing
│       └── db.py                     # `mlxz db backup`, `mlxz db check`
│
├── tests/
│   ├── conftest.py                   # shared fixtures; tiny-model factory; fake-clock
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_residency.py
│   │   ├── test_admission.py         # hypothesis state-machine
│   │   ├── test_block_manager.py     # hypothesis: refcount/leak invariants
│   │   ├── test_prefix_hasher.py
│   │   ├── test_prefix_memory.py     # LRU, byte-budget eviction
│   │   ├── test_prefix_disk.py       # alignment, mmap roundtrip
│   │   ├── test_chunker.py
│   │   ├── test_sampling.py
│   │   ├── test_gguf_bridge.py
│   │   ├── test_api_schemas.py       # OpenAI compat at schema level
│   │   └── test_metrics_registry.py
│   ├── integration/
│   │   ├── test_single_stream_e2e.py # Phase 1 tiny-model end-to-end
│   │   ├── test_openai_client_sdk.py # uses openai-python against fixture server
│   │   ├── test_prefix_cache_hit.py
│   │   ├── test_continuous_batching.py
│   │   ├── test_speculative_lossless.py
│   │   └── test_admission_rejection.py
│   ├── correctness/                  # self-hosted, slower
│   │   ├── test_ppl_wikitext.py
│   │   ├── test_mmlu.py
│   │   ├── test_humaneval.py
│   │   ├── test_niah.py              # needle-in-a-haystack, RULER
│   │   ├── test_prefix_determinism.py # same logits w/ and w/o prefix hit
│   │   └── test_batch_determinism.py  # same output at any batch position
│   ├── soak/
│   │   └── test_48h_agent_replay.py  # nightly
│   └── fixtures/
│       ├── tiny_model/               # 2-layer Llama clone, ~8MB, git-lfs
│       ├── profiles/                 # reference residency plans per chip
│       ├── prompts/                  # standardized benchmark prompts
│       └── agent_traces/             # captured agent workload replays
│
├── benchmarks/
│   ├── run_matrix.py                 # full Phase-0 baseline sweeper
│   ├── agent_replay.py               # locust scenarios
│   ├── compare_to_llama_cpp.py
│   ├── compare_to_mlx_lm.py
│   ├── compare_to_ollama.py
│   ├── plot_results.py               # matplotlib → docs/perf/
│   ├── baseline.json                 # committed; updated only on release tags
│   └── README.md
│
├── scripts/
│   ├── set_wired_limit.sh            # one-liner with safety interlocks
│   ├── download_models.py            # HF hub fetch, SHA-pinned
│   ├── thermal_monitor.sh            # powermetrics wrapper
│   └── ssd_endurance_report.sh       # smartctl → TBW summary
│
├── docs/
│   ├── index.md
│   ├── whitepaper.md                 # the "why" doc
│   ├── implementation-plan.md        # this file
│   ├── architecture.md
│   ├── api-reference.md              # OpenAI-compat surface + extensions
│   ├── quickstart.md
│   ├── tuning-guide.md               # wired-limit, KV bits, prefix cache sizing
│   ├── residency-tables/             # per-hw per-model budgets
│   ├── quantization-recipes.md
│   ├── benchmark-methodology.md
│   ├── how-to-contribute.md
│   └── perf/                         # auto-generated from benchmark runs
│
├── monitoring/
│   ├── grafana-dashboard.json        # importable Grafana dashboard
│   ├── prometheus-alerts.yml         # sample alerting rules
│   └── prometheus-scrape.yml         # sample scrape config snippet
│
├── profiles/                         # committed offline profiles per (chip, model)
│   ├── m3_max_64gb/
│   └── m4_max_64gb/
│
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE                           # Apache-2.0
├── CHANGELOG.md                      # release-please maintained
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── .gitignore
├── .gitattributes                    # git-lfs config for tiny_model
├── .editorconfig
├── .pre-commit-config.yaml
└── mkdocs.yml
```

---

## 3. Core module specifications

Each subsection defines the module's responsibility, its public interface, and the invariants tests bind against.

### 3.1 `mlxz.config`

```python
# src/mlxz/config.py
from pathlib import Path
from typing import Literal
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class KVConfig(BaseModel):
    bits: Literal[4, 8, 16] = 8
    group_size: int = Field(default=64, ge=1, le=256)
    quantized_kv_start: int = Field(default=256, ge=0)    # FP16 for first N tokens
    streaming_sink_size: int = Field(default=4, ge=1)     # Xiao et al. 2023

class PagedConfig(BaseModel):
    block_size: int = Field(default=16, ge=1, le=256)     # tokens per block
    enabled: bool = False            # Phase 3 flips this to True by default

class PrefixCacheConfig(BaseModel):
    memory_budget_gb: float = Field(default=8.0, gt=0)
    disk_budget_gb: float = Field(default=50.0, gt=0)
    disk_path: Path = Path.home() / ".cache/mlxz/prefix"
    disk_tier_enabled: bool = True

    @model_validator(mode="after")
    def include_model_hash_in_path(self) -> Self:
        """Ensures separate disk caches per model to prevent cross-contamination."""
        # Path is appended with model name hash at runtime by the engine
        return self

class SpeculativeConfig(BaseModel):
    enabled: bool = False
    draft_model: str | None = None
    num_draft_tokens: int = Field(default=4, ge=1, le=16)
    max_draft_tokens: int = Field(default=8, ge=1, le=32)
    backoff_threshold: float = Field(default=0.5, gt=0, le=1.0)

    @model_validator(mode="after")
    def draft_tokens_order(self) -> Self:
        assert self.num_draft_tokens <= self.max_draft_tokens, \
            f"num_draft_tokens ({self.num_draft_tokens}) > max_draft_tokens ({self.max_draft_tokens})"
        return self

class SchedulerConfig(BaseModel):
    max_concurrent_requests: int = Field(default=8, ge=1, le=128)
    chunked_prefill_chunk_tokens: int = Field(default=128, ge=1)
    admission_headroom: float = Field(default=0.10, gt=0, le=0.5)

class ServerConfig(BaseModel):
    host: str = "127.0.0.1"          # localhost by default; bind 0.0.0.0 explicitly
    port: int = Field(default=8000, ge=1, le=65535)
    api_key: SecretStr | None = None  # pydantic SecretStr — never serialized to logs/telemetry
    ssl_certfile: Path | None = None  # TLS certificate for non-loopback deployments
    ssl_keyfile: Path | None = None   # TLS private key
    metrics_bind: str = "127.0.0.1:9090"  # separate port for /metrics (never exposed publicly)
    cors_origins: list[str] = Field(default_factory=list)  # empty = CORS disabled
    request_timeout_seconds: float = Field(default=300.0, ge=1.0)  # 5 min max per request

class RuntimeConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MLXZ_", env_nested_delimiter="__",
        toml_file="mlxz.toml",
    )
    model: str                        # HF repo or local path
    draft_model: str | None = None
    wired_limit_mb: int | None = None # None = auto-probe
    kv: KVConfig = Field(default_factory=KVConfig)
    paged: PagedConfig = Field(default_factory=PagedConfig)
    prefix_cache: PrefixCacheConfig = Field(default_factory=PrefixCacheConfig)
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
```

**Invariants.**
- Config is immutable after load. All runtime policy decisions take `RuntimeConfig` by reference.
- No global state. The FastAPI app is constructed via `create_app(config: RuntimeConfig)`.
- Precedence: CLI flags > environment variables > TOML file > defaults.

### 3.2 `mlxz.profile.residency`

Probes `iogpu.wired_limit_mb`, measures actual weight + activation footprints, derives the admission budget.

```python
# src/mlxz/profile/residency.py
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ResidencyBudget:
    wired_limit_bytes: int
    usable_budget_bytes: int          # wired * (1 - headroom)
    weight_bytes: int
    activation_scratch_bytes: int
    kv_budget_bytes: int              # what's left for KV after weights + scratch
    prefix_cache_budget_bytes: int    # memory-tier ceiling

class ResidencyPlanner:
    def probe(self) -> ResidencyBudget: ...
    def plan_for(self, model_bytes: int, cfg: RuntimeConfig) -> ResidencyBudget: ...
    def apply(self, budget: ResidencyBudget) -> None:
        """Calls mx.set_wired_limit; raises ResidencyOverflow with remediation if refused."""

    def project_request_peak(self, input_tokens: int, max_new_tokens: int
                             ) -> int:
        """Used by AdmissionController to decide accept/reject."""
```

**Invariants.**
- `apply()` never exceeds the probed wired cap.
- `ResidencyOverflow` carries a specific remediation string (the exact `sysctl` command) rather than silently degrading.
- `project_request_peak` is monotonic in both arguments.

### 3.3 `mlxz.scheduler.admission`

Pure, deterministic gate between the API layer and the engine. Projects peak memory for each incoming request; accepts or rejects.

```python
# src/mlxz/scheduler/admission.py
from dataclasses import dataclass
from enum import IntEnum

class AdmissionDecision(IntEnum):
    ACCEPT = 0
    REJECT_OVER_BUDGET = 1
    REJECT_QUEUE_FULL = 2
    REJECT_THERMAL = 3
    REJECT_MEMORY_PRESSURE = 4

@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    kv_used_bytes: int
    kv_budget_bytes: int
    running_requests: int
    queued_requests: int
    thermal_state: Literal["normal", "warn", "critical"]
    memory_pressure: Literal["normal", "warn", "critical"]

class AdmissionController:
    def decide(self, request: Request, snap: AdmissionSnapshot
               ) -> tuple[AdmissionDecision, str]:
        """Returns decision + human-readable reason. Deterministic; no I/O."""
```

**Invariants.**
- Monotonicity: if `snap.kv_used_bytes` strictly increases and budget is unchanged, decision never relaxes from REJECT to ACCEPT.
- No preemption. Once accepted, a request runs to completion or client-cancels.
- Rejections return HTTP 429 with `Retry-After` and a JSON body listing projected vs. available resources.

### 3.4 `mlxz.paged_attention.block_manager` (Phase 3)

Fixed-size block pool with reference counting. Block sharing enables zero-copy prefix-cache hits.

```python
# src/mlxz/paged_attention/block_manager.py
from dataclasses import dataclass

@dataclass(slots=True)
class PhysicalBlock:
    idx: int
    refcount: int

class BlockManager:
    def __init__(self, total_blocks: int, block_size: int): ...
    def allocate(self, n_blocks: int) -> list[int]: ...
    def free(self, block_indices: list[int]) -> None: ...
    def incref(self, block_indices: list[int]) -> None: ...
    def copy_on_write(self, block_idx: int) -> int: ...
    @property
    def free_blocks(self) -> int: ...
    @property
    def total_blocks(self) -> int: ...
```

**Invariants (property-tested with Hypothesis).**
- **No leaks.** `free_blocks + sum(refcount > 0 for blocks) == total_blocks` at all times.
- **No double-free.** Freeing a block with refcount > 1 decrements; freeing a block with refcount == 1 returns it to the pool; freeing with refcount == 0 raises.
- **Monotonic refcount.** A block's refcount never goes negative.
- **COW correctness.** After `copy_on_write(b)`, the new block has a deep copy of b's contents and the old block's refcount decreases by 1.

### 3.5 `mlxz.prefix_cache`

Two tiers, two phases. **Phase 2 ships this module first** — before paged attention — with a contiguous-KV implementation. **Phase 3 upgrades it** to block-backed sharing without changing the public API.

```python
# src/mlxz/prefix_cache/base.py
from typing import Protocol

class PrefixCacheProtocol(Protocol):
    async def lookup(self, token_hashes: list[bytes]
                     ) -> tuple[int, CachedPrefix | None]:
        """Returns (n_matched_chunks, cached KV reference)."""
    async def store(self, token_hashes: list[bytes], kv: KVReference) -> None: ...
    def stats(self) -> PrefixCacheStats: ...
```

```python
# src/mlxz/prefix_cache/hasher.py
import hashlib

class RollingPrefixHasher:
    """Emits SHA-256 hashes at every block_size boundary in token stream."""
    def __init__(self, block_size: int): ...
    def hash_chunks(self, tokens: list[int]) -> list[bytes]: ...
```

**Invariants.**
- **Content-addressed correctness.** Two token streams with identical prefixes produce identical hash sequences.
- **Deterministic hit → deterministic logits.** For a prefill that produces next-token logits *L*, a prefix-cache hit followed by resumed prefill produces *L'* with `max |L - L'| < 1e-4`. Enforced in `tests/correctness/test_prefix_determinism.py`.
- **Disk tier alignment.** Every on-disk tensor offset is a multiple of 16384. Fuzzed with Hypothesis.
- **LRU coherence.** Evicting a block at position *p* in the LRU list means every block with access time < *p* has already been evicted.

### 3.6 `mlxz.engine`

Three engines, one interface. Single-stream (Phase 1) → Continuous (Phase 4) → Speculative (Phase 5). All three implement the same abstract engine protocol; the CLI chooses at startup.

```python
# src/mlxz/engine/request.py
from enum import IntEnum

class RequestState(IntEnum):
    QUEUED = 0
    ADMITTED = 1
    PREFILLING = 2
    DECODING = 3
    COMPLETED = 4
    CANCELLED = 5
    REJECTED = 6

@dataclass(slots=True)
class Request:
    id: str
    prompt_tokens: list[int]
    max_tokens: int
    sampling: SamplingParams
    state: RequestState
    output_queue: asyncio.Queue[int | None]  # None sentinel = EOS
    # ... cache pointers, accounting
```

```python
# src/mlxz/engine/continuous.py  (Phase 4)
class ContinuousBatchingEngine:
    async def run(self) -> None:
        """Single compute thread; owns all mx.eval calls."""
        while True:
            # 1. Admit new requests up to budget
            # 2. Classify running requests: prefill chunk vs decode step
            # 3. Build packed batch (prefill chunks first, then decode)
            # 4. Prefix-cache lookup per admitting request
            # 5. Forward pass → sample → update caches
            # 6. Emit tokens to per-request async queues
            # 7. Retire completed requests
```

**Invariants.**
- **Single compute thread.** `mx.eval` is called only from the engine thread; enforced by a thread-identity assertion in an `mx.eval` wrapper.
- **No token loss.** For any request, the sequence of tokens emitted to its output queue equals the sequence produced by a batch=1 reference run (modulo speculative decoding, which is lossless in distribution not in identity).
- **No HOL blocking.** A request with `max_tokens=256` completes within 2× its solo-latency when one `max_tokens=2048` request is in flight.

### 3.7 `mlxz.engine.speculative` (Phase 5)

Vanilla draft-target with Chen et al. rejection sampling. Adaptive draft-token count.

```python
# src/mlxz/engine/speculative.py
def verify_tokenizer_compat(target_tok, draft_tok) -> None:
    """Raise DraftIncompatible with actionable message if vocabs diverge.
    Checked at engine startup — not per request."""
```

**Invariants.**
- **Lossless in distribution.** KL divergence of output distribution vs. non-speculative baseline on a 1000-prompt reference set < 1e-4.
- **Acceptance-rate observability.** Per-request acceptance rate appears in the response `usage` extension and in the `mlxz_speculative_acceptance_rate` Prometheus gauge.
- **Integration with batching.** Within a batch, per-sequence accepted-token counts may differ; the engine handles ragged decode positions correctly.

### 3.8 `mlxz.loader.gguf_bridge`

Reads GGUF using the `gguf` package; applies the K-quant-style mixed-precision recipe (4-bit default, 5-bit on `v_proj`/`down_proj`, 6-bit on embeddings and `lm_head`). Emits MLX-native shards for zero-copy re-load. **No training, no calibration data** — bit-width assignments are fixed per-tensor-name rules.

**Invariant.** WikiText-2 PPL for the converted model is within 0.15 of the GGUF source at the same effective bit-width. Enforced in `tests/correctness/test_ppl_wikitext.py`.

### 3.9 `mlxz.api`

FastAPI app with OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models`. Streaming via SSE. Pydantic request/response models literally mirror OpenAI's JSON schema — so the `openai-python` SDK is a drop-in test fixture.

```python
# src/mlxz/api/openai.py
@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    engine: Annotated[Engine, Depends(get_engine)],
) -> ChatCompletionResponse | StreamingResponse:
    req = Request.from_openai(request)
    decision, reason = engine.scheduler.decide(req, engine.snapshot())
    if decision != AdmissionDecision.ACCEPT:
        raise HTTPException(429, detail={
            "reason": reason,
            "projected_bytes": req.projected_peak_bytes,
            "available_bytes": engine.snapshot().kv_budget_bytes,
        })
    await engine.submit(req)
    if request.stream:
        return StreamingResponse(sse_stream(req), media_type="text/event-stream")
    return await collect_response(req)
```

**Invariants.**
- **OpenAI contract.** Every documented request shape for `openai-python` SDK version ≥1.0 round-trips. Enforced in `tests/integration/test_openai_client_sdk.py`.
- **No internal leakage.** Error responses never include stack traces, internal paths, or model weights metadata beyond what OpenAI returns.
- **Bearer auth optional.** If `config.server.api_key` is set, the server rejects requests missing or mismatching the `Authorization: Bearer ...` header with HTTP 401.

### 3.10 `mlxz.api.metrics`

Prometheus registry, exposed at `/metrics`. Full metric surface:

| Metric | Type | Labels |
|---|---|---|
| `mlxz_requests_total` | Counter | `endpoint`, `status` |
| `mlxz_request_duration_seconds` | Histogram | `endpoint` |
| `mlxz_decode_tokens_per_second` | Histogram | — (no per-request label — avoids cardinality bomb) |
| `mlxz_ttft_seconds` | Histogram | `prefix_cache` (hit/miss) |
| `mlxz_batch_size` | Gauge | — |
| `mlxz_kv_used_bytes` | Gauge | — |
| `mlxz_kv_budget_bytes` | Gauge | — |
| `mlxz_admission_rejections_total` | Counter | `reason` |
| `mlxz_prefix_cache_hits_total` | Counter | `tier` (memory/disk) |
| `mlxz_prefix_cache_hit_bytes_total` | Counter | `tier` |
| `mlxz_speculative_acceptance_rate` | Histogram | — |
| `mlxz_thermal_state` | Gauge (0/1/2) | — |
| `mlxz_rss_bytes` | Gauge | — |
| `mlxz_engine_restarts_total` | Counter | — |
| `mlxz_active_requests` | Gauge | — |

**Important:** No per-request labels on any metric. Per-request attribution (decode speed, acceptance rate) goes to the telemetry DB, not Prometheus. This prevents a cardinality explosion that would OOM the Prometheus client during soak tests.

**Metrics bind address.** `/metrics` is served on a **separate port** (`metrics_bind`, default `127.0.0.1:9090`) from the inference API. This avoids the "should /metrics require auth?" debate — it's simply not exposed on the public interface. Monitoring tools scrape the metrics port directly.

### 3.11 `mlxz.telemetry`

SQLAlchemy 2.x. One row per `Run` (a benchmark invocation or server session). One row per `Request` (OpenAI call). One row per `Measurement` (fine-grained sample).

```python
# src/mlxz/telemetry/models.py
class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    commit_sha: Mapped[str]
    hardware: Mapped[str]                   # "m4_max_64gb"
    model: Mapped[str]
    draft_model: Mapped[str | None]
    quant: Mapped[str]
    kv_bits: Mapped[int]
    wired_limit_mb: Mapped[int]
    config_json: Mapped[str]
    started_at: Mapped[datetime]

class RequestRow(Base):
    __tablename__ = "requests"
    id: Mapped[str] = mapped_column(primary_key=True)     # UUID
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    prompt_tokens: Mapped[int]
    completion_tokens: Mapped[int]
    prefix_cache_hit_tokens: Mapped[int]
    ttft_ms: Mapped[float]
    decode_tps: Mapped[float]
    acceptance_rate: Mapped[float | None]
    rejected_reason: Mapped[str | None]
    created_at: Mapped[datetime]

class Measurement(Base):
    __tablename__ = "measurements"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    sampled_at: Mapped[datetime]
    batch_size: Mapped[int]
    aggregate_decode_tps: Mapped[float]
    kv_used_bytes: Mapped[int]
    rss_bytes: Mapped[int]
    thermal_state: Mapped[str]
```

**Default store.** SQLite at `~/.cache/mlxz/telemetry.db`. Configurable via `MLXZ_TELEMETRY_DSN` env var — point at your WFCA Postgres by setting a libpq URL.

### 3.12 `mlxz.cli`

```
mlxz doctor                      # chip, wired-limit, thermal, MLX version, pass/fail
mlxz serve --model <repo-or-path> [--draft-model ...] [--config mlxz.toml]
mlxz bench --matrix              # Phase 0 baseline sweep
mlxz bench --regression          # CI: compare to baseline.json, exit non-zero on >5% regression
mlxz bench --agent-replay <trace.jsonl>
mlxz convert <src> --out <dst> --recipe q4_km_mlx
mlxz replay <trace.jsonl>        # drives a running server with captured load
```

### 3.13 `mlxz.lifecycle` — Shutdown, Crash Recovery, Supervision

The engine is a long-running event loop on a dedicated thread. Without explicit lifecycle management, crashes, deploys, and SIGTERMs silently drop in-flight requests and leak KV cache.

```python
# src/mlxz/lifecycle/shutdown.py
import asyncio
import signal
from dataclasses import dataclass
from enum import IntEnum

class ServerPhase(IntEnum):
    STARTING = 0       # model loading, cache warm-up
    READY = 1          # accepting requests
    DRAINING = 2       # no new admissions; finishing in-flight
    STOPPED = 3        # all resources released

@dataclass(slots=True)
class DrainResult:
    completed: int
    force_cancelled: int
    drain_duration_seconds: float

class ShutdownCoordinator:
    """Orchestrates graceful shutdown across API, engine, and telemetry layers."""

    def __init__(self, drain_timeout_seconds: float = 30.0):
        self.phase: ServerPhase = ServerPhase.STARTING
        self._drain_timeout = drain_timeout_seconds
        self._shutdown_event = asyncio.Event()

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register SIGTERM and SIGINT handlers. Called once at startup."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._initiate_shutdown)

    def _initiate_shutdown(self) -> None:
        self.phase = ServerPhase.DRAINING
        self._shutdown_event.set()

    async def drain(self, engine: "Engine") -> DrainResult:
        """Wait for running requests to complete, then force-cancel survivors.

        1. Set admission gate to REJECT_SHUTTING_DOWN.
        2. Wait up to drain_timeout for running_requests == 0.
        3. Force-cancel any survivors — emit [DONE] on their SSE streams.
        4. Flush telemetry.
        5. Call mx.metal.clear_cache().
        """
        ...

    @property
    def is_accepting(self) -> bool:
        return self.phase == ServerPhase.READY
```

```python
# src/mlxz/lifecycle/supervisor.py
class EngineThreadSupervisor:
    """Wraps the engine's run() loop. Catches unhandled exceptions,
    sets health to RED, logs the traceback, and optionally restarts."""

    def __init__(self, engine: "Engine", max_restarts: int = 3,
                 restart_backoff_seconds: float = 2.0): ...

    def run_supervised(self) -> None:
        """Target for threading.Thread. Catches exceptions, updates health,
        and restarts the engine loop up to max_restarts times."""
        restarts = 0
        while restarts <= self._max_restarts:
            try:
                self._engine.run()
            except Exception as exc:
                self._health.set_red(reason=f"engine crash: {exc}")
                structlog.get_logger().error("engine_crash",
                    exc_info=True, restart_attempt=restarts)
                restarts += 1
                time.sleep(self._backoff * restarts)
            else:
                break  # clean exit
        if restarts > self._max_restarts:
            structlog.get_logger().critical("engine_max_restarts_exceeded")
            os._exit(1)  # hard exit — don't leave a zombie API server
```

**Invariants.**
- **No silent drops.** Every in-flight SSE stream receives a final `data: [DONE]\n\n` event before the connection closes, even during forced shutdown.
- **Bounded drain.** Shutdown completes within `drain_timeout + 5s` regardless of request state.
- **Crash visibility.** An engine thread crash sets health to RED within 100ms. The health endpoint never reports GREEN when the engine thread is dead.
- **Request journal.** Append-only JSONL at `~/.cache/mlxz/request_journal.jsonl` logs request ID, admission time, and completion/cancellation time. Rotated on clean startup. Enables post-mortem after crashes.

### 3.14 `mlxz.engine.thread_boundary` — Async/Sync Contract

The API layer is async (FastAPI + uvicorn on the main event loop). The engine is synchronous (dedicated compute thread). Crossing this boundary incorrectly causes data corruption, deadlocks, or silent token loss.

```python
# src/mlxz/engine/thread_boundary.py
import janus
import asyncio
from dataclasses import dataclass, field

@dataclass(slots=True)
class RequestBridge:
    """Thread-safe bridge between async API and sync engine.

    Uses janus.Queue for the submission channel (API → engine)
    and per-request janus.Queue for the token channel (engine → API).
    """
    # API thread puts requests here (async side)
    # Engine thread gets requests here (sync side)
    _submit_queue: janus.Queue["Request"] = field(default_factory=lambda: janus.Queue(maxsize=256))

    # Per-request token delivery
    def create_token_channel(self, max_depth: int = 64) -> janus.Queue[int | None]:
        """Returns a janus.Queue. Engine puts tokens on sync side;
        API reads from async side. None sentinel = EOS.
        max_depth provides backpressure — engine pauses decode
        for this request if its channel is full."""
        return janus.Queue(maxsize=max_depth)

    async def submit_async(self, request: "Request") -> None:
        """Called from API thread. Blocks if submission queue is full."""
        await self._submit_queue.async_q.put(request)

    def get_next_sync(self, timeout: float = 0.01) -> "Request | None":
        """Called from engine thread. Non-blocking poll."""
        try:
            return self._submit_queue.sync_q.get_nowait()
        except janus.SyncQueueEmpty:
            return None

class CancellationRegistry:
    """Tracks per-request cancellation events.
    API layer sets the event on client disconnect.
    Engine thread checks each iteration."""

    def __init__(self):
        self._events: dict[str, asyncio.Event] = {}

    def register(self, request_id: str) -> asyncio.Event:
        event = asyncio.Event()
        self._events[request_id] = event
        return event

    def cancel(self, request_id: str) -> None:
        if event := self._events.get(request_id):
            event.set()

    def is_cancelled(self, request_id: str) -> bool:
        if event := self._events.get(request_id):
            return event.is_set()
        return False

    def unregister(self, request_id: str) -> None:
        self._events.pop(request_id, None)
```

**Invariants.**
- **Never use `asyncio.Queue` across threads.** All cross-thread communication uses `janus.Queue`.
- **Backpressure.** If a client stops reading SSE, the per-request token channel fills to `max_depth`, and the engine skips decode steps for that request until space is available. This prevents unbounded memory growth.
- **Cancellation latency.** A client disconnect triggers `CancellationRegistry.cancel()` within one event-loop tick. The engine thread checks `is_cancelled` at the top of each decode iteration. Cancelled requests free their KV cache immediately.
- **No orphaned channels.** `CancellationRegistry.unregister()` and channel cleanup happen in a `finally` block in the SSE stream handler and in `collect_response()`.

### 3.15 `mlxz.security` — Input Validation, Auth, TLS, Supply Chain

Security controls are not a Phase 6 concern. They ship with the first HTTP endpoint.

```python
# src/mlxz/security/limits.py
from pydantic import Field, BaseModel

class RequestLimits(BaseModel):
    """Enforced in Pydantic request schemas before tokenization."""
    max_prompt_tokens: int = Field(default=32768, ge=1, le=131072)
    max_completion_tokens: int = Field(default=4096, ge=1, le=32768)
    max_request_body_bytes: int = Field(default=10_485_760, ge=1)  # 10 MB
    max_concurrent_per_client: int = Field(default=16, ge=1)
    request_timeout_seconds: float = Field(default=300.0, ge=1.0)  # 5 min
```

```python
# src/mlxz/security/auth.py
import hmac
from pydantic import SecretStr
from starlette.middleware.base import BaseHTTPMiddleware

class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Constant-time bearer token validation."""

    def __init__(self, app, api_key: SecretStr | None, exempt_paths: set[str]):
        super().__init__(app)
        self._api_key = api_key
        self._exempt_paths = exempt_paths  # e.g. {"/health/live"}

    async def dispatch(self, request, call_next):
        if self._api_key is None or request.url.path in self._exempt_paths:
            return await call_next(request)
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(token, self._api_key.get_secret_value()):
            return JSONResponse(status_code=401, content={"error": "invalid_api_key"})
        return await call_next(request)
```

```python
# src/mlxz/security/gguf_validator.py
class GGUFValidator:
    """Pre-parse validation layer for untrusted GGUF files.
    Runs BEFORE the gguf parser allocates any tensors."""

    def validate(self, path: Path, max_total_bytes: int | None = None) -> None:
        """Raises GGUFValidationError with specific reason if:
        - File size inconsistent with declared tensor sizes
        - Any tensor shape dimension > 2^24 (16M) or negative
        - Total declared tensor bytes > max_total_bytes (default: 2x physical RAM)
        - File contains non-safetensors tensor format indicators
        """
        ...
```

```python
# src/mlxz/security/config.py (additions to ServerConfig)
class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: SecretStr | None = None       # pydantic SecretStr — never serialized
    ssl_certfile: Path | None = None       # TLS certificate
    ssl_keyfile: Path | None = None        # TLS private key
    metrics_bind: str = "127.0.0.1:9090"   # separate bind for /metrics
    cors_origins: list[str] = Field(default_factory=list)  # empty = no CORS
    request_limits: RequestLimits = Field(default_factory=RequestLimits)
```

**Security middleware stack** (applied in order):
1. **Request body size limit** — Starlette `ContentSizeLimitMiddleware(max_request_body_bytes)`.
2. **Bearer auth** — `BearerAuthMiddleware` with `hmac.compare_digest`.
3. **Rate limiter** — Token-bucket per client IP. Default: 100 req/s burst, 20 req/s sustained.
4. **CORS** — Only if `cors_origins` is non-empty. Never reflects `Origin` header.
5. **Security headers** — `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Cache-Control: no-store`.

**Secrets management.**
- `api_key` and `telemetry_dsn` use `pydantic.SecretStr`. They are excluded from `config_json` serialization in the telemetry recorder.
- `structlog` processors redact fields matching `*key*`, `*secret*`, `*password*`, `*token*`, `*dsn*`.
- The request journal (Section 3.13) never logs prompt content or completion content.
- Auth failures are logged with source IP and timestamp (never the attempted key value).

**Model path validation.**
- Local paths are resolved and must be within a configurable `model_root` (default: `~/.cache/mlxz/models`).
- HF hub downloads only accept `safetensors` format. `HF_ENDPOINT` is pinned to `https://huggingface.co` in the server process.
- GGUF files pass through `GGUFValidator` before the `gguf` parser touches them.

**Invariants.**
- **No plaintext secrets on the wire.** If `api_key` is set and `host != 127.0.0.1` and `ssl_certfile` is None, emit a WARNING at startup: "API key configured without TLS on non-loopback address."
- **No secret leakage.** `SecretStr` fields never appear in logs, telemetry, or error responses. Enforced by a unit test that serializes `RuntimeConfig` and asserts no secret values appear.
- **Input bounds enforced before tokenization.** Request body size is checked by middleware; token limits are validated in the Pydantic schema. The admission controller never sees an unbounded request.

### 3.16 `mlxz.observability` — Structured Logging, Tracing, Request Context

Production debugging requires more than Prometheus counters. Every request needs a correlation ID, and every significant engine event needs a structured log line.

```python
# src/mlxz/observability/context.py
import uuid
import structlog
from contextvars import ContextVar

_request_ctx: ContextVar["RequestContext | None"] = ContextVar("request_ctx", default=None)

@dataclass(frozen=True, slots=True)
class RequestContext:
    request_id: str
    model: str
    prompt_tokens: int
    max_tokens: int
    prefix_cache_hit: bool = False
    created_at: float = field(default_factory=time.monotonic)

    def bind_logger(self) -> structlog.BoundLogger:
        return structlog.get_logger().bind(
            request_id=self.request_id,
            model=self.model,
        )

def new_request_context(**kwargs) -> RequestContext:
    ctx = RequestContext(request_id=str(uuid.uuid4()), **kwargs)
    _request_ctx.set(ctx)
    return ctx
```

**Structured log events** (emitted at the following points):

| Event | When | Fields |
|-------|------|--------|
| `request_admitted` | After admission control passes | `request_id`, `prompt_tokens`, `max_tokens`, `projected_peak_bytes` |
| `request_rejected` | On admission rejection | `request_id`, `reason`, `projected_bytes`, `available_bytes` |
| `prefill_start` | Before prefill forward pass | `request_id`, `prefix_cache_hit`, `hit_tokens`, `remaining_tokens` |
| `prefill_end` | After prefill completes | `request_id`, `prefill_duration_ms`, `ttft_ms` |
| `decode_checkpoint` | Every 64 tokens | `request_id`, `tokens_so_far`, `decode_tps_rolling` |
| `request_completed` | On EOS or max_tokens | `request_id`, `total_tokens`, `total_duration_ms`, `decode_tps` |
| `request_cancelled` | On client disconnect | `request_id`, `tokens_generated`, `reason` |
| `engine_crash` | On unhandled engine exception | `exc_info`, `restart_attempt` |
| `thermal_transition` | On thermal state change | `from_state`, `to_state`, `action` |
| `auth_failure` | On invalid bearer token | `source_ip`, `path` |

**Log configuration.**
- Default level: `INFO` for serving, `DEBUG` for development.
- Per-module level overrides via `MLXZ_LOG_LEVEL_<MODULE>` env vars.
- Request content logging: `MLXZ_LOG_CONTENT=none` (default) | `metadata` (token counts, latency) | `full` (prompt + completion — development only, with startup warning).
- High-frequency events (`decode_checkpoint`) are sampled at 1/64 in production to avoid log flood at 50+ tok/s.

---

## 4. Testing strategy (TDD)

### 4.1 Test tiers

| Tier | Runs where | Runs when | Budget |
|---|---|---|---|
| **Unit** | Hosted (ubuntu-latest + macos-14) | Every PR | < 60 s |
| **Integration (tiny-model)** | Hosted `macos-14` | Every PR | < 5 min |
| **Integration (real-model)** | Self-hosted M4 Max | Every PR to `main`; skipped on forks | < 20 min |
| **Benchmark gate** | Self-hosted M4 Max | PRs labeled `perf`; every tag | < 60 min |
| **Correctness suite** (PPL, MMLU, HumanEval, NIAH, determinism) | Self-hosted M4 Max | Nightly + pre-release | < 6 h |
| **Soak** (48h agent replay) | Self-hosted M4 Max | Weekly + pre-release | 48 h |

### 4.2 Fixtures

- `tests/fixtures/tiny_model/` — committed 2-layer, 4-head, 128-dim Llama-arch model (~8 MB) in git-lfs. Exercises every code path that doesn't require real thermal/memory pressure. Built once with a reproducible seed.
- `tests/fixtures/agent_traces/` — three captured workload shapes in JSONL: `claude_code_day.jsonl` (heavy prefix sharing, bursty), `chat_mixed.jsonl` (low prefix share, steady), `rag_qa.jsonl` (medium prefix share, long contexts).
- `tests/conftest.py` provides: `tiny_model()`, `tiny_tokenizer()`, `mock_memory_monitor()`, `fake_clock()`, `server_process()` (spawns a real server for integration tests).

### 4.3 Property tests (Hypothesis)

The state-machine modules are the primary targets.

```python
# tests/unit/test_block_manager.py
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
from hypothesis import strategies as st

class BlockManagerMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.bm = BlockManager(total_blocks=64, block_size=16)
        self.owned: dict[str, list[int]] = {}

    @rule(seq_id=st.text(min_size=1, max_size=4),
          n=st.integers(min_value=1, max_value=8))
    def allocate(self, seq_id, n):
        if self.bm.free_blocks >= n and seq_id not in self.owned:
            self.owned[seq_id] = self.bm.allocate(n)

    @rule(data=st.data())
    def free(self, data):
        if not self.owned:
            return
        seq_id = data.draw(st.sampled_from(list(self.owned)))
        self.bm.free(self.owned.pop(seq_id))

    @rule(data=st.data())
    def fork(self, data):
        if not self.owned:
            return
        parent = data.draw(st.sampled_from(list(self.owned)))
        new_id = data.draw(st.text(min_size=1, max_size=4))
        if new_id in self.owned:
            return
        self.bm.incref(self.owned[parent])
        self.owned[new_id] = list(self.owned[parent])

    @invariant()
    def no_leaks(self):
        total_owned = sum(len(v) for v in self.owned.values())
        # refcount-adjusted invariant
        assert self.bm.free_blocks + self._accounted_blocks() == self.bm.total_blocks
```

Similar state machines for `AdmissionController` (monotonicity under pressure), `PrefixCacheMemory` (LRU coherence), and `ContinuousBatchingEngine` (no-token-loss under random arrival/cancellation).

### 4.4 Correctness gates (absolute, non-waivable)

| Gate | Threshold | Introduced in |
|---|---|---|
| WikiText-2 PPL drift vs FP16 | ≤ 0.15 | Phase 1 |
| MMLU 5-shot vs published | ≥ published − 1.5 | Phase 1 |
| HumanEval pass@1 vs published | ≥ published − 2.0 | Phase 1 |
| NIAH single-needle @ 32K | ≥ 95% | Phase 1 |
| NIAH multi-needle @ 32K | ≥ 80% | Phase 1 |
| Prefix-cache logit drift (max\|L − L'\|) | < 1e-4 | Phase 2 |
| Batch-position determinism | bytewise equal | Phase 4 |
| Speculative KL vs non-speculative | < 1e-4 | Phase 5 |
| 48h soak memory growth | < 50 MB/h | Phase 6 |
| 48h soak P99 TTFT regression | < 20% from start | Phase 6 |

### 4.5 Performance regression gate

`mlxz bench --regression` compares the current run to `benchmarks/baseline.json`. The baseline is updated only on release tags. Gate:

- Decode tok/s on Llama-3.3-70B Q4 + INT8 KV at 8K context: any regression > 5% fails the PR.
- Aggregate decode tok/s at batch=4: any regression > 5% fails.
- TTFT on warm-prefix 8K request: any regression > 15% fails. (More tolerance because prefix-cache latency has more variance.)
- Thermal-state match: if the runner's thermal state differs by > 5 °C from baseline, the run aborts and reports rather than failing; retried on schedule.

### 4.6 API contract tests

```python
# tests/integration/test_openai_client_sdk.py
import openai

def test_chat_completions_nonstreaming(server_process):
    client = openai.OpenAI(base_url=server_process.url + "/v1", api_key="test")
    resp = client.chat.completions.create(
        model="tiny-model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=16,
    )
    assert resp.choices[0].message.content
    assert resp.usage.completion_tokens <= 16
```

Every major shape of `openai.OpenAI.chat.completions.create` is exercised: streaming, non-streaming, tool-calls pass-through, stop sequences, seed, temperature, logprobs. If the OpenAI SDK adds a new field, the CI catches it when we bump the SDK pin.

---

## 5. CI / CD

### 5.1 `.github/workflows/ci.yml` (hosted)

```yaml
name: ci
on: [pull_request, push]

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - uses: actions/cache@v4
        with:
          path: ~/.cache/uv
          key: uv-lint-${{ runner.os }}-${{ hashFiles('uv.lock') }}
          restore-keys: uv-lint-${{ runner.os }}-
      - run: uv sync --all-extras
      - run: uv lock --check                        # fail if lockfile is stale
      - run: uv run ruff check src tests
      - run: uv run ruff format --check src tests
      - run: uv run pip-audit --strict               # CVE scan on resolved deps

  typecheck:
    runs-on: macos-14                                # pyright must see MLX stubs
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - uses: actions/cache@v4
        with:
          path: ~/.cache/uv
          key: uv-tc-${{ runner.os }}-${{ hashFiles('uv.lock') }}
          restore-keys: uv-tc-${{ runner.os }}-
      - run: uv sync --all-extras
      - run: uv run pyright src

  unit:
    runs-on: macos-14
    timeout-minutes: 15
    strategy:
      matrix: { python: ["3.11", "3.12", "3.13"] }
    steps:
      - uses: actions/checkout@v4
        with: { lfs: true }
      - uses: astral-sh/setup-uv@v3
      - uses: actions/cache@v4
        with:
          path: ~/.cache/uv
          key: uv-unit-${{ runner.os }}-py${{ matrix.python }}-${{ hashFiles('uv.lock') }}
          restore-keys: uv-unit-${{ runner.os }}-py${{ matrix.python }}-
      - run: uv sync --all-extras
      - run: uv run pytest tests/unit tests/integration/test_single_stream_e2e.py tests/integration/test_openai_client_sdk.py -n auto --timeout=30

  unit-linux:
    runs-on: ubuntu-latest                           # platform-independent unit tests
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - uses: actions/cache@v4
        with:
          path: ~/.cache/uv
          key: uv-unit-linux-${{ hashFiles('uv.lock') }}
          restore-keys: uv-unit-linux-
      - run: uv sync --extra dev                     # skip MLX on Linux
      - run: uv run pytest tests/unit -k "not metal" --timeout=30

  docs:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev
      - run: uv run mkdocs build --strict
```

### 5.2 `.github/workflows/bench.yml` (self-hosted)

Triggered on PR label `perf` and on every push to `main`. Emits a commit comment with the delta table.

```yaml
name: bench
on:
  pull_request: { types: [labeled, synchronize] }
  push: { branches: [main] }

permissions:
  contents: read
  pull-requests: write                               # for commit comments

concurrency:
  group: bench-${{ github.ref }}
  cancel-in-progress: true                           # only one bench per ref

jobs:
  bench:
    if: contains(github.event.pull_request.labels.*.name, 'perf') || github.event_name == 'push'
    runs-on: [self-hosted, m4-max, 64gb]
    timeout-minutes: 90
    environment: benchmark                           # requires manual approval for forks
    steps:
      - uses: actions/checkout@v4
        with: { lfs: true }
      - uses: astral-sh/setup-uv@v3
      - uses: actions/cache@v4
        with:
          path: ~/.cache/uv
          key: uv-bench-${{ hashFiles('uv.lock') }}
          restore-keys: uv-bench-
      - run: sudo /usr/local/bin/set-wired-limit 87.5   # restricted sudoers entry
      - run: uv sync --all-extras
      - run: uv run mlxz doctor --strict
      - run: uv run mlxz bench --regression --baseline benchmarks/baseline.json
      - uses: actions/upload-artifact@v4
        with:
          name: bench-results-${{ github.sha }}
          path: ~/.cache/mlxz/telemetry.db
          retention-days: 30
```

**Runner sudoers hardening.** The self-hosted runner's sudoers file restricts sudo to a single command:
```
runner ALL=(root) NOPASSWD: /usr/local/bin/set-wired-limit
```
Where `/usr/local/bin/set-wired-limit` is a copy of `scripts/set_wired_limit.sh` installed during runner provisioning (not from the workflow-checked-out repo). This prevents a malicious PR from modifying the script and executing arbitrary root commands.

### 5.3 `.github/workflows/correctness.yml` (self-hosted, nightly)

```yaml
name: correctness
on:
  schedule: [{ cron: "0 6 * * *" }]
  workflow_dispatch: {}
  pull_request:
    types: [labeled]                                 # also triggered by 'release' label

permissions:
  contents: read

concurrency:
  group: correctness
  cancel-in-progress: false                          # never cancel a running correctness suite

jobs:
  correctness:
    if: >
      github.event_name != 'pull_request' ||
      contains(github.event.pull_request.labels.*.name, 'release')
    runs-on: [self-hosted, m4-max, 64gb]
    timeout-minutes: 420                             # 7 hours
    steps:
      - uses: actions/checkout@v4
        with: { lfs: true }
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --all-extras
      - run: uv run pytest tests/correctness -v --junitxml=correctness.xml --timeout=3600
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: correctness-${{ github.sha }}
          path: correctness.xml
          retention-days: 14
```

### 5.4 `.github/workflows/publish.yml` (PyPI)

```yaml
name: publish
on:
  release:
    types: [published]

permissions:
  id-token: write                                    # OIDC trusted publisher
  contents: read
  attestations: write

jobs:
  publish:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    environment: pypi
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv build
      - uses: pypa/gh-action-pypi-publish@release/v1
      - uses: actions/attest-build-provenance@v1
        with:
          subject-path: dist/*
```

### 5.5 Release flow (release-please)

Conventional Commits drive `release-please` → it opens a release PR whenever `main` advances, tagging minor versions for new features (e.g. `v0.3.0` = Phase 2 complete) and patch versions for fixes.

**Critical gate ordering.** The correctness suite must pass **before** the release PR can be merged, not after. The release PR triggers the correctness workflow via the `release` label. Merging is blocked until the `correctness` status check passes. Only after merge does release-please create the GitHub release and trigger the publish workflow.

```json
// .release-please-manifest.json
{ ".": "0.0.0" }
```

```json
// release-please-config.json
{
  "release-type": "python",
  "packages": {
    ".": {
      "package-name": "mlxz",
      "extra-files": [
        "src/mlxz/__init__.py",
        "pyproject.toml"
      ],
      "changelog-sections": [
        { "type": "feat", "section": "Features" },
        { "type": "fix", "section": "Bug Fixes" },
        { "type": "perf", "section": "Performance" },
        { "type": "docs", "section": "Documentation" }
      ]
    }
  }
}
```

### 5.6 Branch strategy

- **`main`** — always green, always deployable.
- **`phaseN/<slug>`** — long-lived branch per phase (e.g. `phase2/prefix-cache`). Squash-merged into `main` only when phase acceptance criteria pass.
- **Short feature branches** — `feat/<slug>`, `fix/<slug>`, `perf/<slug>`, `docs/<slug>`. Squash-merged. Max lifetime 5 days.
- **No long-running release branches.** `main` is the only line of development. Hotfixes tag off the last release tag only if required for a fielded user.

---

## 6. Benchmarking infrastructure

### 6.1 Reference hardware matrix

| Label | Chip | RAM | SSD | Role |
|---|---|---|---|---|
| `m4-max-64gb` | M4 Max (16c / 40c GPU, 546 GB/s) | 64 GB | 1 TB | Primary reference, CI gate |
| `m3-max-64gb` | M3 Max (14c / 30c, 400 GB/s) | 64 GB | 1 TB | Weekly regression |
| `m3-ultra-192gb` | M3 Ultra (28c / 60c, 819 GB/s) | 192 GB | 2 TB | Headroom tests, high-concurrency |
| `m2-max-32gb` | M2 Max | 32 GB | 512 GB | Downward degradation; Q4 70B must cleanly admission-reject |

### 6.2 Model matrix

| Model | Quant | Size | Purpose |
|---|---|---|---|
| Llama-3.1-8B-Instruct | Q4 | 4.7 GB | Smoke test, high-concurrency reference |
| Gemma-2-9B-it | Q4 | 5.5 GB | GQA variant |
| Qwen-2.5-14B-Instruct | Q4 | 8.5 GB | Mid-tier; KV-quant sensitivity gauge |
| Llama-3.3-70B-Instruct | Q4 | ~40 GB | **Primary flagship target** |
| Qwen-2.5-72B-Instruct | Q4 | ~41 GB | Alt-family 70B-class |
| Llama-3.2-1B-Instruct | Q4 | 0.7 GB | Llama speculative draft |
| Qwen-2.5-0.5B-Instruct | Q4 | 0.4 GB | Qwen speculative draft |

All models pinned by Hugging Face commit SHA in `scripts/download_models.py`. No model weights in the repo; `models.lock.json` commits resolved SHAs.

### 6.3 Prompt / workload suite

- `tests/fixtures/prompts/` — six standardized prompts at 512 / 2K / 8K / 32K tokens × 256-token generation target. Single-stream reference.
- `tests/fixtures/agent_traces/` — three captured replay workloads:
  - `claude_code_day.jsonl` — 500 requests over 4 hours, heavy system-prompt sharing. **Flagship test.**
  - `chat_mixed.jsonl` — 200 requests over 2 hours, minimal prefix sharing.
  - `rag_qa.jsonl` — 100 requests, long context, moderate sharing.

### 6.4 Comparison tooling

`benchmarks/compare_to_llama_cpp.py` shells out to `llama-server` with matched prompts; logs into the same telemetry schema. This is how "we beat llama.cpp by X%" claims are substantiated — always with the exact command that produced both numbers in the PR description.

Similar scripts for `mlx-lm` and `ollama`.

---

## 7. Phased milestones → git tags

| Tag | Phase | Weeks | Deliverable | Acceptance |
|---|---|---|---|---|
| `v0.1.0` | 0 | 1–2 | Baseline bench + `mlxz doctor` + telemetry + governance | llama.cpp / mlx-lm / ollama numbers reproducible within ±5% |
| `v0.2.0` | 1 | 3–4 | Single-stream engine + OpenAI API + residency planner | 70B Q4 at ≥ 12 tok/s; `openai-python` SDK works drop-in; PPL drift ≤ 0.15 |
| `v0.3.0` | 2 | 5–6 | **Content-hashed prefix cache (memory + disk)** | **TTFT ≤ 800 ms on 95%-hit 8K request**; logit drift < 1e-4 |
| `v0.4.0` | 3 | 7–9 | Paged attention + block manager + prefix-cache block upgrade | Single-stream perf unchanged; no block leaks under Hypothesis |
| `v0.5.0` | 4 | 10–11 | Continuous batching + chunked prefill + admission control | **Aggregate ≥ 35 tok/s at batch=4**; 24h sustained test with zero OOMs |
| `v0.6.0` | 5 | 12–13 | Speculative decoding in the batched engine | ≥ 20 tok/s effective on HumanEval; KL < 1e-4 |
| `v1.0.0` | 6 | 14–15 | Hardening + thermal/pressure circuit breakers + Homebrew + docs | 48h soak green; fresh-install → working curl in < 5 min |

**v1.1+** (non-blocking stretch, post v1.0):
- `v1.1.0` — RadixAttention suffix sharing, `/v1/embeddings` endpoint.
- `v1.2.0` — `metal-flash-attention` backend for long-context prefill.
- `v1.3.0` — KIVI INT4 KV behind a flag, with published per-model sensitivity table.
- `v2.0.0-alpha` — MoE support (DeepSeek V3 class), MLX Swift port, ANE-hosted draft (research).

---

## 8. Initial scaffolding files

### 8.1 `pyproject.toml`

```toml
[project]
name = "mlxz"
version = "0.0.0"   # release-please manages
description = "High-throughput local inference server for Apple Silicon"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
authors = [{ name = "Mauricio Zuniga" }]
dependencies = [
    "mlx>=0.22,<0.23",              # pin single minor — 0.x breaks often
    "mlx-lm>=0.21,<0.23",
    "gguf>=0.10,<0.12",             # binary parser — pin upper bound
    "fastapi>=0.115,<1.0",          # 1.0 will break Depends/middleware
    "uvicorn[standard]>=0.30,<1.0",
    "pydantic>=2.7,<3.0",
    "pydantic-settings>=2.4,<3.0",
    "typer>=0.12,<1.0",
    "structlog>=24.0",
    "prometheus-client>=0.20,<1.0",
    "sqlalchemy>=2.0,<3.0",
    "alembic>=1.13,<2.0",
    "huggingface-hub>=0.26",
    "numpy>=1.26,<3.0",
    "httpx>=0.27,<1.0",
    "janus>=1.0,<2.0",              # thread-safe async/sync queues
]

[project.optional-dependencies]
dev = [
    "pytest>=8", "pytest-xdist>=3", "pytest-benchmark>=4", "pytest-asyncio>=0.23",
    "pytest-timeout>=2.3",           # per-test timeout guards
    "hypothesis>=6", "ruff>=0.6", "pyright>=1.1.380",
    "pre-commit>=3.7", "mkdocs-material>=9.5", "mkdocs-mermaid2-plugin>=1.1",
    "openai>=1.40", "locust>=2.31",
    "pip-audit>=2.7",                # CVE scanning in CI
]
postgres = ["psycopg[binary]>=3.2"]

[project.scripts]
mlxz = "mlxz.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pyright]
include = ["src"]
strict = ["src/mlxz"]
pythonVersion = "3.11"

[tool.ruff]
line-length = 100
target-version = "py311"
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF", "N", "PTH", "PL", "ASYNC"]
ignore = ["PLR0913"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --strict-config --timeout=30"
timeout_method = "thread"
asyncio_mode = "auto"
markers = [
  "slow: > 30s runtime",
  "self_hosted: requires Apple Silicon self-hosted runner",
  "correctness: runs full correctness dataset",
  "soak: multi-hour test",
  "metal: requires Apple Metal GPU",
]
```

### 8.2 `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks: [{ id: ruff }, { id: ruff-format }]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks: [{ id: trailing-whitespace }, { id: end-of-file-fixer }, { id: check-toml }, { id: check-yaml }]
  - repo: local
    hooks:
      - id: pyright
        name: pyright
        entry: uv run pyright src
        language: system
        pass_filenames: false
```

### 8.3 `scripts/set_wired_limit.sh`

```bash
#!/usr/bin/env bash
# Raises iogpu.wired_limit_mb to N% of physical RAM. Default 87.5%.
# Reverts automatically at reboot — never persists.
set -euo pipefail
pct="${1:-87.5}"

# Input validation — prevent command injection via python3 -c
if ! [[ "$pct" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "ERROR: percentage must be a number (e.g. 87.5), got: '$pct'" >&2
  exit 1
fi
if (( $(echo "$pct > 100" | bc -l) )) || (( $(echo "$pct <= 0" | bc -l) )); then
  echo "ERROR: percentage must be in (0, 100], got: $pct" >&2
  exit 1
fi

ram_mb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 ))
target=$(python3 -c "import sys; print(int(${ram_mb} * float(sys.argv[1]) / 100))" "$pct")
echo "Setting iogpu.wired_limit_mb=$target (of $ram_mb MB, ${pct}%)"
sudo sysctl "iogpu.wired_limit_mb=$target"
echo "Reverts on reboot. Current: $(sysctl -n iogpu.wired_limit_mb) MB"
```

### 8.4 `CONTRIBUTING.md` — short version

- **Conventional Commits** required (enforced by `commitlint` pre-push hook).
- **One concern per PR.** Feature + unrelated fix = two PRs.
- **Every PR answers three questions in the description:** what benchmark moved, what test proves correctness, what's the rollback plan.
- **No new dependency without justification** in the PR body (size, maintainer health, license, why stdlib or an existing dep can't do it).
- **Every new public API ships with a docstring that includes an invariant and at least one `>>> doctest` example.**
- **Never call `mx.eval` outside the engine thread.** Violations are caught by a runtime assertion; adding a new call site triggers mandatory review from `@CODEOWNERS`.

### 8.5 `.github/PULL_REQUEST_TEMPLATE.md`

```markdown
## Summary
<!-- what changes, why -->

## Benchmark impact
<!-- paste `mlxz bench --regression` output, or N/A with justification -->

## Correctness
<!-- which tests cover this; link to new tests -->

## Rollback
<!-- how we undo this if it regresses silently in 2 weeks -->

## Checklist
- [ ] Conventional Commit title
- [ ] Docstrings + invariants on new public APIs
- [ ] Tests added (unit / integration / correctness as appropriate)
- [ ] No new dependency, or justification present
- [ ] `mlxz doctor` still passes on dev machine
- [ ] No new `mx.eval` call site outside the engine thread
```

---

## 9. Local developer setup (one-liner)

```bash
git clone git@github.com:<owner>/mlxz && cd mlxz
curl -LsSf https://astral.sh/uv/install.sh | sh
git lfs pull                             # tiny test model
uv sync --all-extras
uv run pre-commit install
uv run mlxz doctor
./scripts/set_wired_limit.sh 87.5        # optional, for 70B work
uv run python scripts/download_models.py llama-3.2-1b-q4 llama-3.1-8b-q4
uv run pytest tests/unit                 # should pass in < 60s
uv run mlxz serve --model mlx-community/Llama-3.1-8B-Instruct-4bit &
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b","messages":[{"role":"user","content":"Hi"}]}'
```

---

## 10. Day-one commit sequence

Ordered list of the first ~24 commits, each small and green, to bootstrap the repo from empty to "Phase 0 in progress." Security and lifecycle modules are front-loaded — they ship before the first HTTP endpoint.

1. `chore: initial repo scaffold` — LICENSE (Apache-2.0), README, .gitignore, .gitattributes, .editorconfig
2. `build: pyproject.toml + uv.lock` — dependencies pinned with upper bounds
3. `ci: hosted lint + unit + typecheck + docs workflows` — ruff, pyright on macOS, pytest, pip-audit, mkdocs --strict, uv lock --check
4. `feat(config): RuntimeConfig with pydantic-settings` — includes SecretStr, Field validators, model_validators
5. `feat(types): shared dataclasses, Enums, Protocols`
6. `feat(exceptions): domain error hierarchy`
7. `feat(security): RequestLimits, BearerAuthMiddleware, GGUFValidator, security headers`
8. `feat(observability): RequestContext, structlog config, secret redaction, request journal`
9. `feat(lifecycle): ShutdownCoordinator, EngineThreadSupervisor`
10. `feat(engine): RequestBridge (janus.Queue), CancellationRegistry`
11. `feat(profile): hardware probe (chip, RAM, wired-limit, thermal)`
12. `feat(cli): mlxz doctor` — first working command (including `--smoke`)
13. `test(unit): doctor smoke + hardware-probe + config validation tests`
14. `test(unit): security middleware + auth + limits tests`
15. `feat(telemetry): SQLAlchemy models + Alembic baseline + WAL mode`
16. `feat(api): FastAPI skeleton with /health/{live,ready,startup} + /metrics on separate port`
17. `test(integration): health probes + metrics + auth round-trip`
18. `feat(bench): tiny fixture model + hosted-runner-safe harness`
19. `feat(bench): compare_to_llama_cpp + compare_to_mlx_lm shims`
20. `ci: self-hosted bench workflow (concurrency, timeout, sudoers-restricted)`
21. `ci: nightly correctness workflow + publish workflow skeleton`
22. `feat(monitoring): Grafana dashboard + Prometheus alert rules + scrape config`
23. `docs: whitepaper.md + implementation-plan.md + architecture.md`
24. `chore(release): release-please config + .release-please-manifest.json + v0.1.0-rc0 tag`

At this point the repo has a functional CLI, a passing test suite, security middleware, structured observability, graceful shutdown, an HTTP server skeleton with split health probes and isolated metrics, a working benchmark harness, and all governance in place — ready to begin Phase 1 (single-stream engine + OpenAI endpoints) on a clean Conventional-Commit history.

---

## 11. Risk analysis

**Risk: MLX kernel primitives are insufficient for paged attention.** `mx.fast.SDPA` doesn't accept block-gather indices natively. *Mitigation:* gather into contiguous scratch then call existing SDPA. Quantify the cost in Phase 3; if >10% overhead, add a custom MLX primitive (pure engine work, no training).

**Risk: Prefix cache determinism fails due to FP nondeterminism.** Reduction order in SDPA isn't guaranteed stable. *Mitigation:* pin reduction order at dispatch level; if SDPA won't honor it, fall back to a deterministic reference kernel for the cached-prefix replay check (slower but only in a test path).

**Risk: Continuous batching sees less speedup than CUDA vLLM.** Apple GPUs saturate bandwidth at lower batch sizes. *Mitigation:* measure early in Phase 4 with a batch=2 smoke test; if aggregate gains are <2×, re-scope targets and publish honest numbers. Speculative decoding still applies.

**Risk: FastAPI async + MLX single-thread discipline creates subtle bugs.** Any accidental `mx.eval` from the API thread corrupts state. *Mitigation:* thread-local assertion in every `mx.eval` wrapper; single designated compute thread owns all evals; API layer uses `asyncio.Queue` to hand work over. Code review enforces no new `mx.eval` sites outside the engine.

**Risk: `iogpu.wired_limit_mb` restricted by a future macOS.** *Mitigation:* runtime auto-adjusts to the currently-available limit; admission control scales; publish a downgrade path (Q4 + shorter context) that fits the default unraised cap.

**Risk: Prefix cache disk writes burn SSD endurance.** Agent workloads can churn tens of GB/hour. *Mitigation:* disk tier is opt-in; monitor TBW via `smartctl`; warn at 50% of rated endurance; recommend external Thunderbolt NVMe for heavy use.

**Risk: Apple ships a native serving solution.** Low probability; Apple's AI focus is on-device features. *Mitigation:* OpenAI API surface keeps the project portable — if Apple ships something better, `mlxz` becomes a thin adapter.

**Risk: MLX API churn across 0.x releases.** *Mitigation:* pin MLX minor version; vendor `QuantizedKVCache` and critical files rather than inheriting; CI runs against MLX nightlies to detect breakage early.

**Risk: MLX thread-safety issues (#2067/#2086/#2104/#3078) manifest under real concurrent load.** *Mitigation:* single-compute-thread discipline in code + runtime assertion; 48h soak test catches latent races; subscribe to MLX issues for upstream fixes.

**Risk: Ecosystem churn in draft-model tokenizer compatibility.** Llama-3.2/3.3 share Tiktoken-128k today; a future Llama-4 may orphan the pairing. *Mitigation:* tokenizer-identity probe at load time; maintain a validated-pairings table; plan for self-speculative fallback that doesn't require a tokenizer-compatible draft.

**Risk: Metal 32 KB threadgroup ceiling blocks custom kernels.** *Mitigation:* don't plan on porting kernels that assume Hopper-sized shared memory; stay on `mx.fast.SDPA` + `metal-flash-attention`.

**Risk: The project ships and nobody uses it because it's pure infrastructure.** *Mitigation:* lead with Phase 2's prefix-cache UX win in public communication. "10× faster agents on your Mac" is a stronger pitch than "vLLM for Mac." Pair the v1.0 release with a blog post showing `mlxz` + Aider or Claude Code.

**Risk: Project-level obsolescence from an M5 Ultra with 256 GB.** *Mitigation:* frame the contribution as **serving architecture**, not memory tricks — continuous batching, prefix caching, and admission control remain valuable regardless of bandwidth or RAM. They just become leverage for serving more clients from a single machine instead of squeezing one model onto it.

**Risk: Supply chain attack on the `gguf` dependency.** The `gguf` package parses untrusted binary files with minimal security review. *Mitigation:* pin upper bound (`<0.12`), add `pip-audit` to CI, wrap all `gguf` parsing in `GGUFValidator` that checks declared tensor sizes against physical RAM before allocation. Consider vendoring the reader if it remains small.

**Risk: Engine thread crash leaves API server accepting requests into a dead queue.** *Mitigation:* `EngineThreadSupervisor` catches unhandled exceptions, sets health to RED within 100ms, and restarts the engine up to 3 times with exponential backoff. If max restarts exceeded, the process exits hard (`os._exit(1)`) rather than leaving a zombie HTTP server.

**Risk: Silent data corruption in disk-tier prefix cache.** Partial writes, disk-full, or attacker modification of cache files could inject wrong KV states. *Mitigation:* per-entry SHA-256 checksum validated on load; on mismatch, evict the entry and log a warning. `mlxz db check` CLI command verifies cache integrity.

**Risk: Prometheus cardinality explosion.** Per-request metric labels (`request_id`) create unbounded time series that OOM the Prometheus client. *Mitigation:* no per-request labels on any Prometheus metric. Per-request data goes to the telemetry DB only.

**Risk: Cross-thread `asyncio.Queue` corruption.** Using `asyncio.Queue` between the async API thread and the sync engine thread silently corrupts event loop state. *Mitigation:* all cross-thread communication uses `janus.Queue`. This is a hard architectural invariant enforced by code review and tested under concurrent load.

---

## 12. What this plan deliberately leaves out

- **Serving overflow-mode for models larger than fit.** If Q4 70B doesn't fit with your desired context, run a smaller model or Q4 with less context. Layer streaming and KV disk-spill are not worth the complexity for a batched server.
- **Docker images.** Apple Silicon + virtualized Metal is a minefield; use `uv` and `brew` natively.
- **A web UI.** `curl` and the Prometheus/Grafana pipeline are enough.
- **Fine-tuning integration.** Out of scope — `mlx-lm.lora` and `axolotl` exist.
- **Multi-node distributed inference.** That's `exo`'s lane.
- **Anvil / Agent-Mem integration.** This server is agent-agnostic by design; integration belongs in WFCA's private infra, not here.

---

## 13. Team delegation matrix

Each phase is broken into work streams that can be executed in parallel by specialized agents or team members. Dependencies are explicit — no stream starts until its blockers are complete.

### 13.1 Role definitions

| Role | Responsibility | Tools / Skills |
|---|---|---|
| **Arch** (Architect) | Module interfaces, Protocol definitions, cross-cutting invariants | Plan mode, code review |
| **Engine** | Compute-thread code: forward pass, sampling, cache management, batching | MLX, Metal profiling, pytest-benchmark |
| **API** | FastAPI endpoints, Pydantic schemas, middleware stack, SSE streaming | FastAPI, OpenAI SDK, httpx |
| **Infra** | CI/CD, runner hardening, release automation, packaging, Homebrew | GitHub Actions, uv, hatchling |
| **Security** | Auth, input validation, GGUF validator, secrets management, TLS | OWASP, pip-audit, Hypothesis fuzzing |
| **QA** | Test harness, fixtures, property tests, correctness suite, soak tests | pytest, Hypothesis, locust |
| **Ops** | Monitoring dashboards, alerting rules, telemetry schema, observability | Prometheus, Grafana, structlog |
| **Docs** | Whitepaper, API reference, quickstart, tuning guide, architecture | mkdocs-material, Mermaid |

### 13.2 Phase-by-phase delegation

#### Phase 0 (Weeks 1–2): Scaffold + Baseline

| Stream | Owner | Deliverable | Blocked by |
|--------|-------|-------------|-----------|
| Repo scaffold + config | Arch | Commits 1–6 | — |
| Security middleware | Security | Commits 7, 14 | Config (commit 4) |
| Observability + lifecycle | Ops | Commits 8–10 | Config (commit 4) |
| Hardware profiling + CLI | Engine | Commits 11–13 | Config (commit 4) |
| Telemetry + API skeleton | API | Commits 15–17 | Security (commit 7), Lifecycle (commit 9) |
| Benchmark harness | QA | Commits 18–19 | CLI (commit 12) |
| CI/CD hardening | Infra | Commits 3, 20–21, 24 | — (parallel from day 1) |
| Monitoring + docs | Ops + Docs | Commits 22–23 | API skeleton (commit 16) |

#### Phase 1 (Weeks 3–4): Single-Stream Engine + OpenAI API

| Stream | Owner | Deliverable | Blocked by |
|--------|-------|-------------|-----------|
| SingleStreamEngine | Engine | `engine/single_stream.py`, sampling | Phase 0 complete |
| Model loader (safetensors) | Engine | `loader/safetensors_store.py` | Config |
| GGUF bridge + validator | Security + Engine | `loader/gguf_bridge.py` + validator | GGUFValidator (Phase 0) |
| OpenAI endpoints | API | `/v1/chat/completions`, `/v1/completions`, `/v1/models` | Engine protocol |
| SSE streaming | API | `StreamingResponse` + backpressure | RequestBridge (Phase 0) |
| Residency planner | Engine | `profile/residency.py` | Hardware probe |
| Contract tests | QA | `test_openai_client_sdk.py`, PPL/MMLU/HumanEval gates | Endpoints |
| Correctness baseline | QA | First correctness run, baseline.json | Real model available |

#### Phase 2 (Weeks 5–6): Prefix Cache

| Stream | Owner | Deliverable | Blocked by |
|--------|-------|-------------|-----------|
| Prefix hasher | Engine | `prefix_cache/hasher.py` | — |
| Memory tier | Engine | `prefix_cache/memory.py` | Hasher |
| Disk tier | Engine + Security | `prefix_cache/disk.py` + checksums | Hasher |
| Cache integration | Engine | Wire into SingleStreamEngine | Memory tier |
| Property tests | QA | LRU coherence, disk alignment fuzzing | Memory + disk tiers |
| Determinism gate | QA | `test_prefix_determinism.py` | Cache integration |

#### Phase 3–6: Follow same pattern

Each subsequent phase follows the same delegation structure. The Arch role defines the module protocol first, then Engine and API implement in parallel, Security reviews, QA writes tests, and Infra ensures CI gates pass.

### 13.3 Parallelism rules

1. **Arch defines Protocol first.** No implementation starts until the `Protocol` class and invariants are merged.
2. **Engine and API work in parallel** once the Protocol exists. They integrate via `RequestBridge`.
3. **Security reviews every PR** that touches: model loading, network-facing code, config parsing, or cache I/O.
4. **QA writes tests concurrent with implementation**, not after. Test stubs land in the same PR as the Protocol.
5. **Infra owns CI gate configuration.** Feature teams never modify workflow files directly — they open issues for Infra.
6. **No phase merges to `main` without:** passing correctness suite, security review sign-off, updated docs, and benchmark comparison.

### 13.4 Handoff protocol

When work passes between roles:
1. The outgoing role updates the task status and leaves a comment summarizing what was done, what was deferred, and any known issues.
2. The incoming role reads the task context, reviews the diff, and acknowledges before starting.
3. Blocked tasks are explicitly marked with the blocking task ID — no implicit dependencies.

---

## 14. Next action

Create the GitHub repo (`<owner>/mlxz`, Apache-2.0, public or private is your call), paste the companion `whitepaper.md` and this file into `docs/`, and execute commit #1 from the day-one sequence. Phase 0 baseline numbers should land in `main` within two weeks. Phase 2 (the flagship user-visible UX win — 50–100× TTFT improvement on agent workloads) ships at week 6. Phase 4 (multi-client serving) ships at week 11. v1.0.0 at week 15.

The honest headline claim at v1.0 is: **"`mlxz` serves Llama-3.3-70B on your Mac with 75× faster TTFT for agents, 4× aggregate throughput under load, and a drop-in OpenAI API. No training, no CUDA, no tricks."** Every number is measured against a committed baseline; every regression fails CI; every phase delivers something a user would install the day it ships.
