"""mlxz serve command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from pydantic import SecretStr


def serve(
    model: Annotated[str, typer.Argument(help="HuggingFace repo ID or local path")],
    host: Annotated[str, typer.Option(help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port")] = 8000,
    api_key: Annotated[
        str | None, typer.Option(envvar="MLXZ_API_KEY", help="Bearer auth key")
    ] = None,
    config_file: Annotated[
        Path | None, typer.Option("--config", help="TOML config file")
    ] = None,
) -> None:
    """Start the mlxz inference server."""
    from mlxz.config import RuntimeConfig, ServerConfig
    from mlxz.api.app import create_app

    server_kwargs: dict = {"host": host, "port": port}
    if api_key:
        server_kwargs["api_key"] = SecretStr(api_key)

    config = RuntimeConfig(model=model, server=ServerConfig(**server_kwargs))

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
