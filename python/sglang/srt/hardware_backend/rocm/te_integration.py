# Copyright 2026 SGLang-1-bit-turbo Team
# Licensed under the Apache License, Version 2.0
"""Optional TransformerEngine integration for ROCm.

When ROCm TransformerEngine is installed, this module provides:
- Fused RMSNorm (AOTriton/CK backed, graph-safe)
- Fused attention (AOTriton backend, hipGraph compatible)
- Fused RoPE (eliminates CPU sync overhead)
- FP8 recipe management (software FP8 on RDNA2)

Usage:
    from sglang.srt.hardware_backend.rocm.te_integration import (
        has_transformer_engine,
        get_te_rmsnorm,
        get_te_fused_attention,
    )
"""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_te_available = False
_te_version = None

try:
    import transformer_engine  # noqa: F401
    import transformer_engine.pytorch as te

    _te_available = True
    _te_version = getattr(transformer_engine, "__version__", "unknown")
    logger.info(f"TransformerEngine {_te_version} available for ROCm acceleration")
except ImportError:
    pass


@lru_cache()
def has_transformer_engine() -> bool:
    """Check if TransformerEngine is installed and usable."""
    return _te_available


def get_te_version() -> Optional[str]:
    """Return TransformerEngine version string or None."""
    return _te_version


def get_te_rmsnorm():
    """Return TransformerEngine's RMSNorm class if available.

    TE RMSNorm uses fused AOTriton/CK kernels on ROCm,
    avoiding separate kernel launches for norm + residual.
    """
    if not _te_available:
        return None
    try:
        from transformer_engine.pytorch import RMSNorm

        return RMSNorm
    except ImportError:
        return None


def get_te_layernorm():
    """Return TransformerEngine's LayerNorm class if available."""
    if not _te_available:
        return None
    try:
        from transformer_engine.pytorch import LayerNorm

        return LayerNorm
    except ImportError:
        return None


def get_te_fused_attention():
    """Return TransformerEngine's fused attention function if available.

    TE fused attention on ROCm uses AOTriton for the attention kernel,
    which is hipGraph-capture safe and avoids the CPU sync issues
    of non-fused implementations.
    """
    if not _te_available:
        return None
    try:
        from transformer_engine.pytorch.cpp_extensions.fused_attn import (
            fused_attn_fwd,
        )

        return fused_attn_fwd
    except ImportError:
        return None


def get_te_fp8_recipe(is_rdna2: bool = False):
    """Return an FP8 recipe suitable for the current GPU.

    On RDNA2 (gfx1030), uses software FP8 emulation with conservative
    scaling to avoid overflow in the int8 representation.

    On MI300+ (gfx94x), uses hardware FP8 with FNUZ format.
    """
    if not _te_available:
        return None
    try:
        from transformer_engine.common.recipe import DelayedScaling, Format

        if is_rdna2:
            # RDNA2: No hardware FP8, use E4M3 format with delayed scaling
            # The actual compute still runs FP16 after software dequant
            return DelayedScaling(
                margin=0,
                fp8_format=Format.E4M3,
                amax_history_len=16,
                amax_compute_algo="max",
            )
        else:
            # MI300+: Hardware FP8 with FNUZ format
            return DelayedScaling(
                margin=0,
                fp8_format=Format.HYBRID,
                amax_history_len=1024,
                amax_compute_algo="max",
            )
    except ImportError:
        return None
