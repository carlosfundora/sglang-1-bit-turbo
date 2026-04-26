"""ROCm architecture helpers for Triton attention tuning."""

from __future__ import annotations

import torch

from sglang.srt.utils import is_hip

_CACHED_GCN_ARCH: str | None = None


def get_gcn_arch() -> str:
    """Return the current ROCm GCN architecture name, cached after first lookup."""
    global _CACHED_GCN_ARCH
    if _CACHED_GCN_ARCH is None:
        if is_hip():
            try:
                _CACHED_GCN_ARCH = torch.cuda.get_device_properties(0).gcnArchName
            except Exception:
                _CACHED_GCN_ARCH = ""
        else:
            _CACHED_GCN_ARCH = ""
    return _CACHED_GCN_ARCH


def is_gfx1030() -> bool:
    """Return True for RDNA2 gfx103x devices."""
    arch = get_gcn_arch()
    return "gfx103" in arch
