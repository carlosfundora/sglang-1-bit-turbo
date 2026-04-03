from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_is_hip = is_hip()


@cache_once
def _jit_clamp_position_module(dtype: torch.dtype) -> Module:
    """Compile and cache the JIT clamp_position module for a given dtype."""
    args = make_cpp_args(dtype)
    return load_jit(
        "clamp_position",
        *args,
        cuda_files=["elementwise/clamp_position.cuh"],
        cuda_wrappers=[
            ("clamp_position", f"ClampPosition<{args}>::run"),
        ],
    )


def clamp_position_cuda(seq_lens: torch.Tensor) -> torch.Tensor:
    """Compute positions = clamp(seq_lens - 1, min=0) on CUDA.

    Supported dtypes: torch.int32, torch.int64.
    """
    if _is_hip:
        # ROCm does not ship the tvm_ffi-backed JIT helper in this environment.
        # Keep HIP on the native tensor fallback so baseline serving can proceed.
        return torch.clamp(seq_lens - 1, min=0).to(seq_lens.dtype)

    dst = torch.empty_like(seq_lens)
    module = _jit_clamp_position_module(seq_lens.dtype)
    module.clamp_position(dst, seq_lens)
    return dst
