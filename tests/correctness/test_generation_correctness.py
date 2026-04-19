"""Correctness checks for generation primitives used by serving engines."""
from __future__ import annotations

import mlx.core as mx

from mlxz.engine.request import StopChecker
from mlxz.engine.sampling import sample
from mlxz.types import SamplingParams


def test_stop_checker_detects_cross_token_boundary() -> None:
    checker = StopChecker(["</END>"])
    assert checker.check("Hello </")[0] is False
    should_stop, matched = checker.check("END> world")
    assert should_stop is True
    assert matched == "</END>"


def test_sampling_is_seed_reproducible() -> None:
    logits = mx.array([0.1, 0.2, 0.3, 0.4])
    params = SamplingParams(temperature=0.8, top_p=1.0, seed=1234)

    def run_sequence() -> list[int]:
        key = mx.random.key(params.seed)
        tokens: list[int] = []
        for _ in range(32):
            key = mx.random.split(key)[0]
            tok, _ = sample(logits, params, key)
            tokens.append(tok)
        return tokens

    assert run_sequence() == run_sequence()
