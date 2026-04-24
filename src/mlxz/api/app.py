"""FastAPI application factory for mlxz.

:func:`create_app` is the single entry point for constructing the inference
server.  It wires the security middleware stack, includes routers, and
configures a lifespan context manager for startup/shutdown orchestration.
"""

from __future__ import annotations

import os
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import mlxz
from mlxz.api.health import HealthState, _health_state, router as health_router
from mlxz.api.openai import router as openai_router
from mlxz.security import (
    BearerAuthMiddleware,
    ContentSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from mlxz.types import ServerPhase

if TYPE_CHECKING:
    from mlxz.config import RuntimeConfig

logger = structlog.get_logger()

# Paths exempt from bearer-token authentication.
_AUTH_EXEMPT_PATHS: set[str] = {
    "/health",
    "/health/live",
    "/health/ready",
    "/health/startup",
}


def _parse_bind(bind: str) -> tuple[str, int]:
    """Parse host:port bind string."""
    host, sep, port_str = bind.rpartition(":")
    if sep == "" or not host:
        raise ValueError(f"Invalid bind address: {bind!r}")
    return host, int(port_str)


def create_app(config: RuntimeConfig) -> FastAPI:
    """Build and return a fully-configured :class:`FastAPI` application.

    Parameters
    ----------
    config:
        Validated runtime configuration.  Drives middleware settings,
        CORS origins, and server behaviour.

    Returns
    -------
    FastAPI
        The application instance, ready to be served by uvicorn.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Manage startup and shutdown lifecycle.

        On startup: load model, create engine, start engine thread.
        On shutdown: drain requests, stop engine, join thread.
        """
        logger.info("server_starting", model=config.model)

        # --- Startup ---
        from mlxz.engine.continuous import ContinuousBatchingEngine
        from mlxz.engine.single_stream import SingleStreamEngine
        from mlxz.engine.speculative import SpeculativeEngine
        from mlxz.engine.thread_boundary import CancellationRegistry, RequestBridge
        from mlxz.loader.safetensors_store import ModelStore
        from mlxz.profile.hardware import detect_hardware
        from mlxz.profile.residency import ResidencyPlanner
        from mlxz.scheduler.admission import AdmissionController
        from mlxz.telemetry.db import create_engine_from_config
        from mlxz.telemetry.recorder import TelemetryRecorder
        from mlxz.api.metrics import create_metrics_app
        from mlxz.attention import patch_attention_memory_efficient_threshold
        import uvicorn

        # 1. Load model
        store = ModelStore()
        model, tokenizer, weight_bytes = store.load(config.model)
        _health_state.load_progress = 0.5
        patch_attention_memory_efficient_threshold(
            config.attention.memory_efficient_threshold
        )

        # 2. Plan residency budget
        planner = ResidencyPlanner()
        budget = planner.plan_for(weight_bytes, config)

        # 3. Create engine — select based on runtime mode
        bridge = RequestBridge()
        cancellations = CancellationRegistry()
        use_speculative = config.speculative.enabled
        use_continuous = config.scheduler.max_concurrent_requests > 1
        if use_speculative:
            draft_model_path = config.speculative.draft_model or config.draft_model
            if draft_model_path is None:
                raise ValueError(
                    "speculative.enabled=true requires speculative.draft_model "
                    "or draft_model to be set"
                )
            engine = SpeculativeEngine(config, bridge, cancellations)
            draft_model, draft_tokenizer, _ = store.load(draft_model_path)
            engine.set_draft_model(draft_model, draft_tokenizer)
            logger.info(
                "engine_mode",
                mode="speculative",
                draft_model=draft_model_path,
                draft_k=config.speculative.num_draft_tokens,
            )
        elif use_continuous:
            engine = ContinuousBatchingEngine(config, bridge, cancellations)
            logger.info("engine_mode", mode="continuous_batching",
                        max_batch=config.scheduler.max_concurrent_requests)
        else:
            engine = SingleStreamEngine(config, bridge, cancellations)
            logger.info("engine_mode", mode="single_stream")
        engine.set_model(model, tokenizer)
        engine.set_budget(budget)

        # Set up prefix cache
        from mlxz.prefix_cache.memory import PrefixCacheMemory
        from mlxz.prefix_cache.disk import PrefixCacheDisk

        memory_cache = PrefixCacheMemory(
            memory_budget_bytes=int(config.prefix_cache.memory_budget_gb * 1024**3)
        )
        disk_cache = None
        if config.prefix_cache.disk_tier_enabled:
            import hashlib

            model_hash = hashlib.sha256(config.model.encode()).hexdigest()[:16]
            disk_cache = PrefixCacheDisk(
                disk_path=config.prefix_cache.disk_path,
                disk_budget_bytes=int(config.prefix_cache.disk_budget_gb * 1024**3),
                model_hash=model_hash,
            )
        engine.set_prefix_cache(
            memory_cache=memory_cache,
            disk_cache=disk_cache,
            block_size=config.prefix_cache.block_size,
        )

        # 4. Create admission controller
        n_layers, n_heads, head_dim = engine.model_arch
        admission = AdmissionController(
            budget,
            config,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
        )

        # 5. Store on app.state
        app.state.engine = engine
        app.state.admission = admission
        app.state.tokenizer = tokenizer
        app.state.cancellations = cancellations
        app.state.bridge = bridge

        # 6. Start engine thread
        engine_thread = threading.Thread(
            target=engine.run, name="mlxz-engine", daemon=True
        )
        engine_thread.start()

        # 7. Start metrics server on separate bind address
        metrics_server = None
        metrics_thread = None
        try:
            metrics_host, metrics_port = _parse_bind(config.server.metrics_bind)
            metrics_server = uvicorn.Server(
                uvicorn.Config(
                    create_metrics_app(),
                    host=metrics_host,
                    port=metrics_port,
                    log_level="warning",
                    access_log=False,
                )
            )
            metrics_thread = threading.Thread(
                target=metrics_server.run,
                name="mlxz-metrics",
                daemon=True,
            )
            metrics_thread.start()
            logger.info("metrics_server_started", bind=config.server.metrics_bind)
        except Exception:
            logger.warning(
                "metrics_server_start_failed",
                bind=config.server.metrics_bind,
                exc_info=True,
            )

        # 8. Start telemetry recorder
        telemetry = None
        telemetry_run_id = None
        try:
            telemetry = TelemetryRecorder(create_engine_from_config())
            hw = detect_hardware()
            telemetry_run_id = telemetry.start_run(
                config,
                hardware=f"{hw.chip_name}|{hw.memory_gb}GB",
                commit_sha=os.environ.get("GITHUB_SHA", "local"),
            )
            logger.info("telemetry_started", run_id=telemetry_run_id)
        except Exception:
            logger.warning("telemetry_start_failed", exc_info=True)

        app.state.telemetry = telemetry
        app.state.telemetry_run_id = telemetry_run_id

        # 9. Mark as ready
        _health_state.phase = ServerPhase.READY
        _health_state.engine_alive = True
        _health_state.load_progress = 1.0
        logger.info("server_ready")

        yield

        # --- Shutdown ---
        logger.info("server_shutting_down")
        _health_state.phase = ServerPhase.DRAINING
        _health_state.engine_alive = False

        await engine.shutdown()
        engine_thread.join(timeout=5)
        bridge.close()

        if metrics_server is not None:
            metrics_server.should_exit = True
        if metrics_thread is not None:
            metrics_thread.join(timeout=5)

        if telemetry is not None and telemetry_run_id is not None:
            try:
                telemetry.end_run(telemetry_run_id)
            finally:
                telemetry.close(timeout=5.0)

        _health_state.phase = ServerPhase.STOPPED
        logger.info("server_stopped")

    app = FastAPI(
        title="mlxz",
        version=mlxz.__version__,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Include routers
    # ------------------------------------------------------------------
    app.include_router(health_router)
    app.include_router(openai_router)

    # ------------------------------------------------------------------
    # Middleware stack
    #
    # Middleware is applied in reverse order of addition (last added
    # executes first).  The desired execution order on an incoming
    # request is:
    #
    #   1. ContentSizeLimitMiddleware  (reject oversized bodies first)
    #   2. BearerAuthMiddleware        (authenticate)
    #   3. SecurityHeadersMiddleware   (add response headers)
    #   4. CORSMiddleware              (optional, CORS preflight)
    #
    # Therefore we add them in reverse: CORS, SecurityHeaders, Auth,
    # ContentSizeLimit.
    # ------------------------------------------------------------------

    # 4. CORS (optional -- only if origins are configured).
    if config.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 3. Security response headers.
    app.add_middleware(SecurityHeadersMiddleware)

    # 2. Bearer token authentication.
    app.add_middleware(
        BearerAuthMiddleware,
        api_key=config.server.api_key,
        exempt_paths=_AUTH_EXEMPT_PATHS,
    )

    # 1. Body size limit (outermost -- runs first).
    app.add_middleware(
        ContentSizeLimitMiddleware,
        max_bytes=config.server.request_limits.max_request_body_bytes,
    )

    return app
