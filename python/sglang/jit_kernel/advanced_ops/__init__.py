"""
Advanced operation optimizations ported from gfxATOM (optional).
- moe_optimized: Mixture of Experts with fused operations
- sparse_attn_v4: Sparse attention patterns
- topK: Efficient top-K selection
- split_chunk: Memory-efficient chunking
"""

from . import (
    moe_optimized,
    sparse_attn_v4,
    split_chunk,
    topK,
)

__all__ = [
    "moe_optimized",
    "sparse_attn_v4",
    "split_chunk",
    "topK",
]
