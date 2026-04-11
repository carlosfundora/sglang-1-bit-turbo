# Copyright 2026 SGLang-1-bit-turbo Team
# Licensed under the Apache License, Version 2.0
"""Optimized server defaults for RDNA2 (gfx1030) GPUs.

All RDNA2 variants (gfx1030-1036) target gfx1030 for widest compatibility:
  - RX 6700 XT (gfx1031): 40 CUs, 12 GB VRAM, 192-bit bus
  - RX 6800/6900 XT (gfx1030): 60-80 CUs, 16 GB VRAM, 256-bit bus
  - Wave32 execution (not Wave64 like MI-series)
  - No FP8 matrix cores (software FP8 dequant only)
  - No WMMA hardware (RDNA3+ only)
  - hipGraph supported but needs explicit enablement
  - HSA_OVERRIDE_GFX_VERSION=10.3.0 required for gfx1030 compat
"""

import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_gfx1031_server_defaults() -> Dict[str, Any]:
    """Return optimized ServerArgs defaults for gfx1031.

    These are applied when gfx1031 is detected and no explicit
    overrides are provided by the user.
    """
    return {
        # Attention: AITER backend with Triton paths for RDNA2.
        # CK kernels are bypassed: decode uses unified_attention (Triton),
        # extend uses extend_attention_fwd (SGLang Triton with Wave32 tuning).
        "attention_backend": "aiter",

        # CUDA graphs work on HIP but need smaller batch sizes for 12GB VRAM
        "disable_cuda_graph": False,
        "cuda_graph_bs": [1, 2, 4, 8, 16, 32],

        # Memory: conservative for 12GB VRAM
        "mem_fraction_static": 0.80,

        # Chunked prefill: smaller chunks for RDNA2 throughput
        "chunked_prefill_size": 2048,

        # Triton tuning hints
        "_triton_num_warps": 2,       # Wave32 → fewer warps than Wave64
        "_triton_num_stages": 2,      # Conservative pipelining
        "_triton_block_m": 64,        # Tuned for 40 CUs
        "_triton_block_n": 64,
    }


def apply_gfx1031_env() -> None:
    """Set environment variables required for gfx1031 compatibility.

    Called early in server initialization when gfx1031 is detected.
    Does NOT override variables already set by the user.
    """
    env_defaults = {
        # Core compatibility: all RDNA2 chips use gfx1030 base ISA
        "HSA_OVERRIDE_GFX_VERSION": "10.3.0",

        # Architecture targeting — gfx1030 for widest RDNA2 compatibility
        "PYTORCH_ROCM_ARCH": "gfx1030",
        "AMDGPU_TARGETS": "gfx1030",
        "HIP_VISIBLE_DEVICES": "0",

        # RDNA2 kernel dispatch + gfxGRAPH activation gate
        "SGLANG_RDNA2_KERNELS": "1",

        # Triton tuning for RDNA2
        "TRITON_PRINT_AUTOTUNING": "0",

        # hipGraph enablement
        "HIP_GRAPH_ENABLED": "1",

        # Memory allocator tuning for 12GB VRAM
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",

        # Disable features that don't work on RDNA2
        "SGLANG_DISABLE_FLASHINFER": "1",  # FlashInfer is CUDA-only

        # AITER: force Triton paths (bypass CK kernels that need CDNA)
        "SGLANG_USE_AITER": "1",
        "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    }

    applied = []
    for key, value in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(f"{key}={value}")

    if applied:
        logger.info(f"gfx1031 env defaults applied: {', '.join(applied)}")


def get_triton_wave32_config(op_type: str = "attention") -> Dict[str, int]:
    """Return Triton kernel configuration tuned for Wave32 RDNA2.

    Wave32 GPUs have half the threads per wavefront compared to Wave64 (CDNA).
    This affects optimal block sizes and warp counts.

    Args:
        op_type: One of 'attention', 'rmsnorm', 'rope', 'softmax', 'gemm'
    """
    configs = {
        "attention": {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "num_warps": 2,
            "num_stages": 2,
            "waves_per_eu": 0,
        },
        "rmsnorm": {
            "BLOCK_SIZE": 1024,
            "num_warps": 4,
            "num_stages": 1,
        },
        "rope": {
            "BLOCK_S": 32,
            "num_warps": 2,
            "num_stages": 1,
        },
        "softmax": {
            "BLOCK_SIZE": 1024,
            "num_warps": 4,
            "num_stages": 1,
        },
        "gemm": {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "num_warps": 4,
            "num_stages": 2,
        },
    }
    return configs.get(op_type, configs["attention"])


def get_optimal_batch_sizes(vram_gb: float = 12.0) -> list:
    """Return optimal CUDA graph batch sizes for the given VRAM.

    For 12 GB VRAM (RX 6700 XT), we use smaller batch sizes to
    avoid OOM during graph capture.
    """
    if vram_gb <= 8:
        return [1, 2, 4, 8]
    elif vram_gb <= 12:
        return [1, 2, 4, 8, 16, 32]
    elif vram_gb <= 16:
        return [1, 2, 4, 8, 16, 32, 64]
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128]
