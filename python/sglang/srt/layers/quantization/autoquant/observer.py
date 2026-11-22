"""AutoQuant observer — per-layer KV tensor statistics sampler.

The observer accumulates statistics on K and V tensors as they pass
through the KV pool. Stats are used by the (future) calibration
explorer to score each codec candidate per layer.

Never blocks the hot path. Sampling is rate-limited and falls
silently to a no-op on any error.

Stats captured per layer per side (K/V):
  - dynamic_range  (max(abs(x)))
  - mean_abs       (E[|x|])
  - rms            (sqrt(E[x^2]))
  - kurtosis       (peakedness — high → bad for uniform quant)
  - sparsity       (fraction of |x| < eps)
  - sample_count   (how many tensors observed)

Usage::

    from sglang.srt.layers.quantization.autoquant.observer import TensorStatsObserver

    obs = TensorStatsObserver(n_layers=32, sample_every=64)

    # in the KV write hot path:
    obs.observe(layer_id, side="K", x=k_tensor)
    obs.observe(layer_id, side="V", x=v_tensor)

    # at calibration end:
    snapshot = obs.snapshot()                  # plain dict, JSON-safe
    obs.dump_json("/tmp/autoquant_stats.json")
"""

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class _SideStats:
    """Running stats for one (layer, side) pair. EMA-based to bound memory."""

    sample_count: int = 0
    dynamic_range: float = 0.0     # running max
    mean_abs_ema: float = 0.0      # EMA of E[|x|]
    rms_ema: float = 0.0           # EMA of sqrt(E[x^2])
    kurtosis_ema: float = 0.0      # EMA of E[((x-mu)/sigma)^4]
    sparsity_ema: float = 0.0      # EMA of frac(|x| < eps)
    last_observed_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "dynamic_range": self.dynamic_range,
            "mean_abs": self.mean_abs_ema,
            "rms": self.rms_ema,
            "kurtosis": self.kurtosis_ema,
            "sparsity": self.sparsity_ema,
            "last_observed_at": self.last_observed_at,
        }


class TensorStatsObserver:
    """Per-layer KV tensor statistics observer.

    Parameters
    ----------
    n_layers
        Total number of decoder layers in the model.
    sample_every
        Only sample 1 tensor every N observations. Default 64. Set to 1
        for full sampling (calibration mode), higher for live mode.
    ema_alpha
        EMA decay factor for running stats. Default 0.05 (slow average).
    sparsity_eps
        Threshold below which |x| is considered "sparse". Default 1e-4.
    enabled
        Master kill-switch. When False the observer is a complete no-op.
    """

    def __init__(
        self,
        n_layers: int,
        sample_every: int = 64,
        ema_alpha: float = 0.05,
        sparsity_eps: float = 1e-4,
        enabled: bool = True,
    ):
        self.n_layers = n_layers
        self.sample_every = max(1, int(sample_every))
        self.ema_alpha = float(ema_alpha)
        self.sparsity_eps = float(sparsity_eps)
        self.enabled = bool(enabled)

        # _stats[(layer_id, side)] = _SideStats
        self._stats: Dict[tuple, _SideStats] = {}
        self._counter = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def observe(self, layer_id: int, side: str, x) -> None:
        """Sample stats from KV tensor `x` for (layer_id, side='K'|'V').

        Best-effort. Never raises; failures degrade to a no-op.
        """
        if not self.enabled:
            return
        try:
            # Rate-limit: only every Nth call actually computes stats.
            with self._lock:
                self._counter += 1
                do_sample = (self._counter % self.sample_every) == 0
            if not do_sample:
                return
            self._update(layer_id, side, x)
        except Exception as e:  # never poison the hot path
            logger.debug("AutoQuant observer: sample failed: %s", e)

    def _update(self, layer_id: int, side: str, x) -> None:
        # Local imports — avoid forcing torch into module load if observer
        # is never used.
        import torch

        if not isinstance(x, torch.Tensor):
            return
        if x.numel() == 0:
            return

        # Cheap path: detach + flatten + cast → float32 on the same device.
        with torch.no_grad():
            t = x.detach()
            if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                t = t.float()
            else:
                t = t.float()
            tabs = t.abs()

            # Use .item() at the end to extract scalars; cheap for small N.
            dyn = float(tabs.max().item())
            mean_abs = float(tabs.mean().item())
            rms = float(t.pow(2).mean().sqrt().item())
            sparsity = float((tabs < self.sparsity_eps).float().mean().item())
            mu = float(t.mean().item())
            sigma = max(rms, 1e-8)
            # kurtosis ~ E[((x-mu)/sigma)^4]
            kurt = float(((t - mu) / sigma).pow(4).mean().item())

        key = (int(layer_id), str(side))
        with self._lock:
            s = self._stats.get(key)
            if s is None:
                s = _SideStats(
                    sample_count=1,
                    dynamic_range=dyn,
                    mean_abs_ema=mean_abs,
                    rms_ema=rms,
                    kurtosis_ema=kurt,
                    sparsity_ema=sparsity,
                    last_observed_at=time.time(),
                )
                self._stats[key] = s
                return

            s.sample_count += 1
            s.dynamic_range = max(s.dynamic_range, dyn)
            a = self.ema_alpha
            s.mean_abs_ema = (1 - a) * s.mean_abs_ema + a * mean_abs
            s.rms_ema = (1 - a) * s.rms_ema + a * rms
            s.kurtosis_ema = (1 - a) * s.kurtosis_ema + a * kurt
            s.sparsity_ema = (1 - a) * s.sparsity_ema + a * sparsity
            s.last_observed_at = time.time()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Return a JSON-safe snapshot of all observed stats."""
        with self._lock:
            return {
                "n_layers": self.n_layers,
                "sample_every": self.sample_every,
                "ema_alpha": self.ema_alpha,
                "sparsity_eps": self.sparsity_eps,
                "total_observations": self._counter,
                "layers": {
                    f"L{layer_id}_{side}": s.to_dict()
                    for (layer_id, side), s in self._stats.items()
                },
            }

    def dump_json(self, path: str) -> None:
        """Write `snapshot()` as pretty-printed JSON to `path`."""
        try:
            with open(path, "w") as f:
                json.dump(self.snapshot(), f, sort_keys=True, indent=2)
            logger.info("AutoQuant observer: dumped %d obs to %s",
                        self._counter, path)
        except Exception as e:
            logger.warning("AutoQuant observer: dump_json failed (%s): %s",
                           path, e)

    def reset(self) -> None:
        """Clear all stats."""
        with self._lock:
            self._stats.clear()
            self._counter = 0
