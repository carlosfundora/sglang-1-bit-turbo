"""
ROCm/RDNA2 Triton scatter-store kernel for KV cache.

Replaces the Python-level `k_cache[indices] = k` fallback used when the CUDA
JIT store_cache kernel is unavailable on HIP.  Each program handles one token
row, loading from the flat k/v tensors and scatter-storing into the KV cache.

RDNA2 safety: `tl.arange(0, BLOCK)` produces exactly `BLOCK` elements.  When
`BLOCK > row_dim` (non-power-of-2 row_dim), inactive column lanes (offs >=
row_dim) would compute OOB addresses on RDNA2 (which validates VA for all
wavefront lanes, even exec-masked ones).  We clamp with `offs_safe =
tl.where(mask, offs, 0)` so every lane accesses the valid row-start address.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_kv_cache_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    row_dim: int,
    BLOCK: tl.constexpr,
):
    """Fused scatter-store: k[pid,:] → k_cache[indices[pid],:] (same for v).

    All tensors are 2D contiguous flat views: [N, row_dim] / [cache_size, row_dim].
    """
    pid = tl.program_id(0).to(tl.int64)
    slot = tl.load(indices_ptr + pid).to(tl.int64)

    offs = tl.arange(0, BLOCK)
    mask = offs < row_dim
    # ROCm/RDNA2: clamp inactive column lanes to index 0 (valid row-start addr).
    offs_safe = tl.where(mask, offs, 0)

    k_row = tl.load(k_ptr + pid * row_dim + offs_safe, mask=mask, other=0.0)
    tl.store(k_cache_ptr + slot * row_dim + offs_safe, k_row, mask=mask)

    v_row = tl.load(v_ptr + pid * row_dim + offs_safe, mask=mask, other=0.0)
    tl.store(v_cache_ptr + slot * row_dim + offs_safe, v_row, mask=mask)


def store_kv_cache_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,
) -> None:
    """Scatter-store K and V rows into the KV cache using a Triton kernel.

    Args:
        k: Source key tensor, any shape whose product equals N * row_dim.
        v: Source value tensor, same constraint.
        k_cache: Destination key cache [cache_size, ...], flattened to 2D.
        v_cache: Destination value cache [cache_size, ...], flattened to 2D.
        indices: Cache slot indices for each of the N source rows, shape [N].
        row_dim: Number of elements per row (tp_k_head_num * qk_head_dim).
    """
    N = indices.numel()
    if N == 0:
        return

    k_flat = k.reshape(N, row_dim).contiguous()
    v_flat = v.reshape(N, row_dim).contiguous()
    k_cache_flat = k_cache.reshape(-1, row_dim)
    v_cache_flat = v_cache.reshape(-1, row_dim)

    BLOCK = triton.next_power_of_2(row_dim)
    # On RDNA2 (wave32): scale warps so each thread handles ~16 elements.
    # 1 warp = 32 threads; BLOCK/32 elements per thread with 1 warp.
    # Double to 2 warps once per-thread work shrinks below 16 (BLOCK >= 512).
    num_warps = 2 if BLOCK >= 512 else 1

    _store_kv_cache_kernel[(N,)](
        k_flat,
        v_flat,
        k_cache_flat,
        v_cache_flat,
        indices,
        row_dim=row_dim,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
