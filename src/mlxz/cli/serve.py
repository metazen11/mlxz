"""mlxz serve command."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from pydantic import SecretStr


def serve(
    model: Annotated[str, typer.Argument(help="HuggingFace repo ID or local path")],
    host: Annotated[str, typer.Option(help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port")] = 8000,
    max_concurrent_requests: Annotated[
        int | None,
        typer.Option(
            "--max-concurrent-requests",
            min=1,
            help="Override scheduler concurrency (set to 1 for single-stream baseline)",
        ),
    ] = None,
    api_key: Annotated[
        str | None, typer.Option(envvar="MLXZ_API_KEY", help="Bearer auth key")
    ] = None,
    config_file: Annotated[
        Path | None, typer.Option("--config", help="TOML config file")
    ] = None,
) -> None:
    """Start the mlxz inference server."""
    from mlxz.config import RuntimeConfig, SchedulerConfig, ServerConfig
    from mlxz.api.app import create_app

    config_kwargs: dict = {"model": model}
    if config_file is not None:
        if not config_file.exists():
            raise typer.BadParameter(f"Config file not found: {config_file}")
        with config_file.open("rb") as f:
            loaded = tomllib.load(f)
        if not isinstance(loaded, dict):
            raise typer.BadParameter("Config file must contain a top-level table")
        config_kwargs.update(loaded)

    server_from_file = config_kwargs.get("server", {})
    if not isinstance(server_from_file, dict):
        raise typer.BadParameter("'server' section in config must be a table")

    scheduler_from_file = config_kwargs.get("scheduler", {})
    if not isinstance(scheduler_from_file, dict):
        raise typer.BadParameter("'scheduler' section in config must be a table")

    server_kwargs: dict = {
        **server_from_file,
        "host": host,
        "port": port,
    }
    if api_key:
        server_kwargs["api_key"] = SecretStr(api_key)
    config_kwargs["server"] = ServerConfig(**server_kwargs)

    scheduler_kwargs: dict = dict(scheduler_from_file)
    if max_concurrent_requests is not None:
        scheduler_kwargs["max_concurrent_requests"] = max_concurrent_requests
    config_kwargs["scheduler"] = SchedulerConfig(**scheduler_kwargs)

    config = RuntimeConfig(**config_kwargs)

    app = create_app(config)

    ssl_kwargs: dict = {}
    if config.server.ssl_certfile:
        ssl_kwargs["ssl_certfile"] = str(config.server.ssl_certfile)
    if config.server.ssl_keyfile:
        ssl_kwargs["ssl_keyfile"] = str(config.server.ssl_keyfile)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info",
        **ssl_kwargs,
    )
