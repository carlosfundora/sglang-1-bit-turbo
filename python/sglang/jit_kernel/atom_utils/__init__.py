"""
ATOM-RS atom.utils package.
Provides real CPU/GPU buffer management and re-exports from submodules.
"""

import numpy as np
import torch
from typing import Optional

from sglang.jit_kernel.atom_utils.forward_context import (
    AttnState,
    AttentionMetaData,
    Context,
    ForwardContext,
    get_forward_context,
    set_forward_context,
    reset_forward_context,
)

# =============================================================================
# CpuGpuBuffer — real implementation matching the API used by backends.py
#
# Usage:
#   buf = CpuGpuBuffer(size, dtype=torch.int32, device='cuda')
#   buf = CpuGpuBuffer(rows, cols, dtype=torch.int32, device='cuda')
#   buf.np[:]         → numpy view of the pinned CPU allocation
#   buf.cpu           → pinned CPU torch.Tensor
#   buf.copy_to_gpu() → copies entire buffer to GPU, returns GPU tensor
#   buf.copy_to_gpu(n)→ copies first n elements to GPU, returns GPU tensor
# =============================================================================

class CpuGpuBuffer:
    """Pinned CPU buffer with on-demand copy-to-GPU semantics."""

    def __init__(self, *shape, dtype: torch.dtype = torch.float32,
                 device: str = "cuda"):
        if not shape:
            raise ValueError("CpuGpuBuffer requires at least one size argument")
        self._shape = tuple(shape)
        self._dtype = dtype
        self._device = device

        # Allocate pinned-memory CPU tensor and numpy view
        try:
            self._cpu = torch.empty(self._shape, dtype=dtype, pin_memory=True)
        except Exception:
            self._cpu = torch.empty(self._shape, dtype=dtype)
        self._np = self._cpu.numpy()

        # GPU mirror (allocated on first copy_to_gpu call)
        self._gpu: Optional[torch.Tensor] = None

    @property
    def cpu(self) -> torch.Tensor:
        return self._cpu

    @property
    def np(self) -> np.ndarray:
        return self._np

    @property
    def gpu(self) -> Optional[torch.Tensor]:
        return self._gpu

    def copy_to_gpu(self, n: Optional[int] = None) -> torch.Tensor:
        """Copy first *n* elements (or all) from CPU to GPU and return the GPU tensor."""
        if n is None:
            src = self._cpu
        else:
            src = self._cpu.reshape(-1)[:n]

        out = src.to(self._device, non_blocking=True)
        self._gpu = out
        return out

    def copy_from_gpu(self) -> None:
        """Copy GPU tensor back to pinned CPU buffer."""
        if self._gpu is not None:
            self._cpu.copy_(self._gpu.cpu())
            self._np = self._cpu.numpy()

    def __repr__(self):
        return (f"CpuGpuBuffer(shape={self._shape}, dtype={self._dtype}, "
                f"device={self._device})")


__all__ = [
    "CpuGpuBuffer",
    "AttnState",
    "AttentionMetaData",
    "Context",
    "ForwardContext",
    "get_forward_context",
    "set_forward_context",
    "reset_forward_context",
]
