"""Residency planner -- wired-limit probing and memory budget derivation.

Probes ``iogpu.wired_limit_mb`` via sysctl, measures weight and activation
footprints, and derives the admission budget that the scheduler uses to
accept or reject requests.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from mlxz.exceptions import ResidencyOverflow
from mlxz.types import ResidencyBudget

if TYPE_CHECKING:
    from mlxz.config import RuntimeConfig

# Fraction of the wired limit reserved as OS/system headroom.
_DEFAULT_HEADROOM: float = 0.10

# Conservative per-layer activation scratch estimate (bytes).
_ACTIVATION_SCRATCH_PER_LAYER: int = 8 * 1024 * 1024  # 8 MB


def _sysctl_int(key: str) -> int | None:
    """Read an integer sysctl value, returning ``None`` on failure."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


class ResidencyPlanner:
    """Derives a :class:`ResidencyBudget` from probed hardware limits.

    The planner is intentionally stateless -- call :meth:`probe` or
    :meth:`plan_for` to get an immutable budget snapshot.
    """

    def __init__(self, headroom: float = _DEFAULT_HEADROOM) -> None:
        if not 0.0 < headroom < 1.0:
            msg = f"headroom must be in (0, 1), got {headroom}"
            raise ValueError(msg)
        self._headroom = headroom

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(self) -> ResidencyBudget:
        """Probe the wired limit and return a budget with zero weight/KV.

        This is the "how much room do we have?" query before any model
        is loaded.
        """
        wired_limit_mb = _sysctl_int("iogpu.wired_limit_mb")
        if not wired_limit_mb:
            # Fall back to total physical memory minus headroom.
            hw_memsize = _sysctl_int("hw.memsize")
            if hw_memsize is None:
                msg = (
                    "Cannot determine memory limit: both iogpu.wired_limit_mb "
                    "and hw.memsize are unavailable"
                )
                raise ResidencyOverflow(
                    msg,
                    remediation=(
                        "Run: sudo sysctl iogpu.wired_limit_mb=<MB>"
                    ),
                )
            wired_limit_bytes = hw_memsize
        else:
            wired_limit_bytes = wired_limit_mb * 1024 * 1024

        usable = int(wired_limit_bytes * (1.0 - self._headroom))

        return ResidencyBudget(
            wired_limit_bytes=wired_limit_bytes,
            usable_budget_bytes=usable,
            weight_bytes=0,
            activation_scratch_bytes=0,
            kv_budget_bytes=usable,
            prefix_cache_budget_bytes=0,
        )

    def plan_for(
        self,
        model_bytes: int,
        cfg: RuntimeConfig,
    ) -> ResidencyBudget:
        """Calculate a full residency budget for the given model.

        Parameters
        ----------
        model_bytes:
            Total weight footprint in bytes (post-quantisation).
        cfg:
            The runtime configuration; used for ``wired_limit_mb`` override
            and prefix-cache budget.
        """
        # Determine wired limit.
        if cfg.wired_limit_mb is not None:
            wired_limit_bytes = cfg.wired_limit_mb * 1024 * 1024
        else:
            probed = self.probe()
            wired_limit_bytes = probed.wired_limit_bytes

        usable = int(wired_limit_bytes * (1.0 - self._headroom))
        activation_scratch = _ACTIVATION_SCRATCH_PER_LAYER  # simplified

        prefix_budget_bytes = int(
            cfg.prefix_cache.memory_budget_gb * 1024**3
        )

        remaining = usable - model_bytes - activation_scratch - prefix_budget_bytes
        if remaining < 0:
            deficit_mb = abs(remaining) // (1024 * 1024)
            current_mb = wired_limit_bytes // (1024 * 1024)
            needed_mb = current_mb + deficit_mb + 512  # buffer
            raise ResidencyOverflow(
                f"Model requires {model_bytes / 1024**3:.1f} GB but only "
                f"{usable / 1024**3:.1f} GB usable budget available",
                remediation=(
                    f"Run: sudo sysctl iogpu.wired_limit_mb={needed_mb}"
                ),
            )

        return ResidencyBudget(
            wired_limit_bytes=wired_limit_bytes,
            usable_budget_bytes=usable,
            weight_bytes=model_bytes,
            activation_scratch_bytes=activation_scratch,
            kv_budget_bytes=remaining,
            prefix_cache_budget_bytes=prefix_budget_bytes,
        )

    def project_request_peak(
        self,
        input_tokens: int,
        max_new_tokens: int,
        kv_bits: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
    ) -> int:
        """Estimate peak KV-cache bytes for a single request.

        The estimate is monotonic in both ``input_tokens`` and
        ``max_new_tokens`` -- increasing either never decreases the
        projected peak.

        Parameters
        ----------
        input_tokens:
            Number of prompt tokens.
        max_new_tokens:
            Maximum generation length.
        kv_bits:
            KV quantisation width (4, 8, or 16).
        n_layers:
            Number of transformer layers.
        n_heads:
            Number of KV heads (after GQA grouping).
        head_dim:
            Dimension per attention head.

        Returns
        -------
        int
            Projected peak bytes for keys *and* values combined.
        """
        total_tokens = input_tokens + max_new_tokens
        bytes_per_element = kv_bits / 8.0
        # 2x for keys + values
        peak = int(
            2 * total_tokens * n_layers * n_heads * head_dim * bytes_per_element
        )
        return peak
