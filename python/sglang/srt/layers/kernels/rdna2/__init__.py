"""RDNA2 (gfx1030) optimized HIP kernels for SGLang.

This module provides Wave32-tuned HIP kernels compiled via:
  1. AITER's @compile_ops JIT system (preferred — fastest path)
  2. torch.utils.cpp_extension inline compilation (fallback)
  3. Pure Triton kernels (universal fallback)

All kernels are tuned for RDNA2's 32-thread wavefronts and lack of
dedicated matrix cores. Launch configs use 2-4 warps (64-128 threads)
with vec8 memory access patterns for bandwidth-optimal execution.

Supported operations:
  - RMSNorm (fused add variant included)
  - FP8 software dequantization (int8 → fp16 with scale)
  - RoPE positional encoding (NeoX and GPT-J styles)
  - Activation functions (SiLU, GELU)
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Detect if we're on RDNA2
_is_rdna2 = False
try:
    from sglang.srt.hardware_backend.rocm.arch_detection import is_rdna2

    _is_rdna2 = is_rdna2()
except ImportError:
    # Fallback: check env
    _is_rdna2 = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "").startswith("10.3")

# Try AITER JIT first, then torch cpp_extension, then Triton
_backend = "none"
_aiter_available = False
_torch_ext_available = False

try:
    from aiter.jit.core import compile_ops

    _aiter_available = True
    _backend = "aiter"
    logger.info("RDNA2 kernels: using AITER JIT backend")
except ImportError:
    try:
        from torch.utils.cpp_extension import load_inline

        _torch_ext_available = True
        _backend = "torch_ext"
        logger.info("RDNA2 kernels: using torch cpp_extension backend")
    except ImportError:
        _backend = "triton"
        logger.info("RDNA2 kernels: using Triton fallback backend")


def get_backend() -> str:
    """Return the active kernel compilation backend."""
    return _backend


def is_available() -> bool:
    """Return True if optimized RDNA2 kernels can be compiled."""
    return _backend in ("aiter", "torch_ext")


# Wave32 launch constants
WAVE_SIZE = 32
RDNA2_WARPS = 4  # 4 warps × 32 threads = 128 threads/block
RDNA2_BLOCK = WAVE_SIZE * RDNA2_WARPS  # 128
RDNA2_VEC_WIDTH = 8  # vec8 loads for bandwidth
