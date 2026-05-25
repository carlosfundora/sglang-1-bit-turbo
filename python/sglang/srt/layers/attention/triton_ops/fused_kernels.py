"""
Fused Triton kernels from gfxATOM for improved performance.
- QKV+Norm+RoPE fusion
- Sigmoid+Mul+Quant fusion (with FP8)
- RMSNorm variants

These kernels are pure Triton with minimal dependencies.
"""

import sys
import os

# Add jit_kernel path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

FUSED_KERNELS_AVAILABLE = {}

# Import kernels that work without AITER-specific types
try:
    from sglang.jit_kernel.triton_fused_qkv_norm_rope_cache import (
        triton_fused_norm_rope_cache,
    )
    FUSED_KERNELS_AVAILABLE['triton_fused_norm_rope_cache'] = True
except ImportError as e:
    FUSED_KERNELS_AVAILABLE['triton_fused_norm_rope_cache'] = False
    print(f"[WARNING] triton_fused_norm_rope_cache not available: {e}")

try:
    from sglang.jit_kernel.triton_fused_sigmoid_mul_quant import (
        fused_sigmoid_mul_fp8_quant,
    )
    FUSED_KERNELS_AVAILABLE['fused_sigmoid_mul_fp8_quant'] = True
except ImportError as e:
    FUSED_KERNELS_AVAILABLE['fused_sigmoid_mul_fp8_quant'] = False
    print(f"[WARNING] fused_sigmoid_mul_fp8_quant not available: {e}")

try:
    from sglang.jit_kernel.triton_gemma_rmsnorm import (
        gemma_rmsnorm_triton,
    )
    FUSED_KERNELS_AVAILABLE['gemma_rmsnorm_triton'] = True
except ImportError as e:
    FUSED_KERNELS_AVAILABLE['gemma_rmsnorm_triton'] = False
    print(f"[WARNING] gemma_rmsnorm_triton not available: {e}")

try:
    from sglang.jit_kernel.triton_rmsnorm_nw import (
        rmsnorm_nw,
    )
    FUSED_KERNELS_AVAILABLE['rmsnorm_nw'] = True
except ImportError as e:
    FUSED_KERNELS_AVAILABLE['rmsnorm_nw'] = False
    print(f"[WARNING] rmsnorm_nw not available: {e}")

# Provide dummy functions for graceful fallback
if not FUSED_KERNELS_AVAILABLE['triton_fused_norm_rope_cache']:
    def triton_fused_norm_rope_cache(*args, **kwargs):
        raise RuntimeError("triton_fused_norm_rope_cache kernel not available")

if not FUSED_KERNELS_AVAILABLE['fused_sigmoid_mul_fp8_quant']:
    def fused_sigmoid_mul_fp8_quant(*args, **kwargs):
        raise RuntimeError("fused_sigmoid_mul_fp8_quant kernel not available")

if not FUSED_KERNELS_AVAILABLE['gemma_rmsnorm_triton']:
    def gemma_rmsnorm_triton(*args, **kwargs):
        raise RuntimeError("gemma_rmsnorm_triton kernel not available")

if not FUSED_KERNELS_AVAILABLE['rmsnorm_nw']:
    def rmsnorm_nw(*args, **kwargs):
        raise RuntimeError("rmsnorm_nw kernel not available")

__all__ = [
    "FUSED_KERNELS_AVAILABLE",
    "triton_fused_norm_rope_cache",
    "fused_sigmoid_mul_fp8_quant",
    "gemma_rmsnorm_triton",
    "rmsnorm_nw",
]
