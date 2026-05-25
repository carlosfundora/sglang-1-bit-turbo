"""
Standalone gfxATOM utilities for ATOM-RS.
Refactored to remove circular dependencies and gfxATOM-specific infrastructure.
"""

import torch
from typing import Optional, Dict, Any, List

# =============================================================================
# Core Buffer Management (replaces atom.utils.CpuGpuBuffer)
# =============================================================================

class CpuGpuBuffer:
    """CPU-GPU buffer for efficient tensor movement."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.device_id = 0 if device == "cuda" else None
    
    def allocate(self, shape: tuple, dtype: torch.dtype):
        """Allocate buffer on device."""
        return torch.empty(shape, dtype=dtype, device=self.device)
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to device."""
        return tensor.to(self.device)
    
    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to CPU."""
        return tensor.cpu()


# =============================================================================
# Forward Context Classes (replaces atom.utils.forward_context)
# =============================================================================

class AttentionMetaData:
    """Metadata for attention operations."""
    
    def __init__(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        num_heads: int = 32,
        head_dim: int = 128,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
    
    def get_q_shape(self) -> tuple:
        return (self.batch_size, self.seq_len, self.num_heads, self.head_dim)
    
    def get_kv_shape(self) -> tuple:
        return (self.batch_size, self.seq_len, self.num_heads, self.head_dim)


class AttnState:
    """Attention state container."""
    
    def __init__(self):
        self.past_key_values = []
        self.attention_mask = None


class Context:
    """Request context for model inference."""
    
    def __init__(self):
        self.request_id = 0
        self.seq_len = 128
        self.device = "cuda"
        self.metadata = AttentionMetaData()


# =============================================================================
# Block Convert Utilities (replaces atom.utils.block_convert)
# =============================================================================

def kv_indices_generate_triton(
    seq_len: int,
    block_size: int = 128,
):
    """Generate KV cache block indices for Triton kernels."""
    num_blocks = (seq_len + block_size - 1) // block_size
    indices = torch.arange(num_blocks, device="cuda")
    return indices


# =============================================================================
# TBO Utils (replaces atom.utils.tbo.ubatch_splitting)
# =============================================================================

class UBatchSlice:
    """Micro-batch slice for attention splitting."""
    
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
    
    def __len__(self) -> int:
        return self.end - self.start


def split_attn_metadata(
    metadata: AttentionMetaData,
    num_slices: int = 4,
) -> List[AttentionMetaData]:
    """Split attention metadata into micro-batches."""
    batch_size = metadata.batch_size
    slice_size = max(1, batch_size // num_slices)
    
    slices = []
    for i in range(0, batch_size, slice_size):
        slice_meta = AttentionMetaData(
            batch_size=min(slice_size, batch_size - i),
            seq_len=metadata.seq_len,
            num_heads=metadata.num_heads,
            head_dim=metadata.head_dim,
        )
        slices.append(slice_meta)
    
    return slices


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'CpuGpuBuffer',
    'AttentionMetaData',
    'AttnState',
    'Context',
    'kv_indices_generate_triton',
    'UBatchSlice',
    'split_attn_metadata',
]
