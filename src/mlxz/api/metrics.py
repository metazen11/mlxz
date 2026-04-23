"""Prometheus metrics registry and standalone metrics application for mlxz.

All metrics follow the naming convention ``mlxz_<subsystem>_<stat>`` and
deliberately avoid per-request labels to prevent cardinality explosions.
Per-request attribution is handled by the telemetry DB, not Prometheus.

The metrics application is designed to run on a separate port
(``config.server.metrics_bind``) and should never be exposed on the
public inference interface.
"""

from __future__ import annotations

from fastapi import FastAPI, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Request-level counters and histograms
requests_total = Counter(
    "mlxz_requests_total",
    "Total HTTP requests handled by the inference API.",
    labelnames=["endpoint", "status"],
)

request_duration_seconds = Histogram(
    "mlxz_request_duration_seconds",
    "End-to-end request latency in seconds.",
    labelnames=["endpoint"],
)

# Decode performance
decode_tokens_per_second = Histogram(
    "mlxz_decode_tokens_per_second",
    "Observed decode throughput per completed request (tokens/s).",
)

# Time to first token
ttft_seconds = Histogram(
    "mlxz_ttft_seconds",
    "Time to first token in seconds.",
    labelnames=["prefix_cache"],
)

# Scheduler / engine state
batch_size = Gauge(
    "mlxz_batch_size",
    "Current continuous-batching batch size.",
)

kv_used_bytes = Gauge(
    "mlxz_kv_used_bytes",
    "Current KV cache memory usage in bytes.",
)

kv_budget_bytes = Gauge(
    "mlxz_kv_budget_bytes",
    "Total KV cache memory budget in bytes.",
)

# Admission
admission_rejections_total = Counter(
    "mlxz_admission_rejections_total",
    "Total requests rejected by the admission controller.",
    labelnames=["reason"],
)

# Prefix cache
prefix_cache_hits_total = Counter(
    "mlxz_prefix_cache_hits_total",
    "Total prefix cache hits.",
    labelnames=["tier"],
)

prefix_cache_hit_bytes_total = Counter(
    "mlxz_prefix_cache_hit_bytes_total",
    "Total bytes served from prefix cache hits.",
    labelnames=["tier"],
)

# Speculative decoding
speculative_acceptance_rate = Histogram(
    "mlxz_speculative_acceptance_rate",
    "Speculative decoding acceptance rate per batch (0.0 to 1.0).",
)

# System-level gauges
thermal_state = Gauge(
    "mlxz_thermal_state",
    "Current thermal state (0=normal, 1=warn, 2=critical).",
)

rss_bytes = Gauge(
    "mlxz_rss_bytes",
    "Resident set size of the server process in bytes.",
)

engine_restarts_total = Counter(
    "mlxz_engine_restarts_total",
    "Total engine restart events.",
)

active_requests = Gauge(
    "mlxz_active_requests",
    "Number of in-flight inference requests.",
)


# ---------------------------------------------------------------------------
# Standalone metrics FastAPI application
# ---------------------------------------------------------------------------


def create_metrics_app() -> FastAPI:
    """Build a minimal FastAPI app that serves ``/metrics`` for Prometheus.

    This app is intended to be run on a separate port
    (``config.server.metrics_bind``), isolated from the public inference API.
    """
    app = FastAPI(
        title="mlxz-metrics",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        """Serve Prometheus text exposition format."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app
