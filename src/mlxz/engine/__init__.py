"""Inference engine implementations."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlxz.engine.continuous import ContinuousBatchingEngine
    from mlxz.engine.draft import DraftModel
    from mlxz.engine.request import Request, Token
    from mlxz.engine.sampling import sample
    from mlxz.engine.single_stream import SingleStreamEngine
    from mlxz.engine.speculative import SpeculativeEngine
    from mlxz.engine.thread_boundary import CancellationRegistry, MxEvalGuard, RequestBridge

__all__ = [
    "CancellationRegistry",
    "ContinuousBatchingEngine",
    "DraftModel",
    "MxEvalGuard",
    "Request",
    "RequestBridge",
    "SingleStreamEngine",
    "SpeculativeEngine",
    "Token",
    "sample",
]


def __getattr__(name: str) -> Any:
    if name == "ContinuousBatchingEngine":
        from mlxz.engine.continuous import ContinuousBatchingEngine

        return ContinuousBatchingEngine
    if name == "DraftModel":
        from mlxz.engine.draft import DraftModel

        return DraftModel
    if name == "Request":
        from mlxz.engine.request import Request

        return Request
    if name == "Token":
        from mlxz.engine.request import Token

        return Token
    if name == "sample":
        from mlxz.engine.sampling import sample

        return sample
    if name == "SingleStreamEngine":
        from mlxz.engine.single_stream import SingleStreamEngine

        return SingleStreamEngine
    if name == "SpeculativeEngine":
        from mlxz.engine.speculative import SpeculativeEngine

        return SpeculativeEngine
    if name == "CancellationRegistry":
        from mlxz.engine.thread_boundary import CancellationRegistry

        return CancellationRegistry
    if name == "MxEvalGuard":
        from mlxz.engine.thread_boundary import MxEvalGuard

        return MxEvalGuard
    if name == "RequestBridge":
        from mlxz.engine.thread_boundary import RequestBridge

        return RequestBridge
    raise AttributeError(name)
