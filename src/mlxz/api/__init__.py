"""mlxz API package — FastAPI application factory and routers."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlxz.api.app import create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from mlxz.api.app import create_app

        return create_app
    raise AttributeError(name)
