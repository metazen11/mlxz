"""FastAPI application factory for mlxz.

:func:`create_app` is the single entry point for constructing the inference
server.  It wires the security middleware stack, includes routers, and
configures a lifespan context manager for startup/shutdown orchestration.
"""

from __future__ import annotations

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
        from mlxz.engine.request import Request
        from mlxz.engine.single_stream import SingleStreamEngine
        from mlxz.engine.continuous import ContinuousBatchingEngine
        from mlxz.engine.thread_boundary import CancellationRegistry, RequestBridge
        from mlxz.loader.safetensors_store import ModelStore
        from mlxz.profile.residency import ResidencyPlanner
        from mlxz.scheduler.admission import AdmissionController

        # 1. Load model
        store = ModelStore()
        model, tokenizer, weight_bytes = store.load(config.model)
        _health_state.load_progress = 0.5

        # 2. Plan residency budget
        planner = ResidencyPlanner()
        budget = planner.plan_for(weight_bytes, config)

        # 3. Create engine — select based on max_concurrent_requests
        bridge = RequestBridge()
        cancellations = CancellationRegistry()
        use_continuous = config.scheduler.max_concurrent_requests > 1
        if use_continuous:
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
        engine.set_prefix_cache(memory_cache=memory_cache, disk_cache=disk_cache)

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

        # 7. Mark as ready
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
