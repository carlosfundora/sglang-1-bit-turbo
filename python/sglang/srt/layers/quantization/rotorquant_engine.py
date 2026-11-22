"""RotorQuant KV cache quantization engine for SGLang.

Implements PlanarQuant (2D Givens rotation) and IsoQuant (4D quaternion rotation)
from the RotorQuant paper. These are data-oblivious quantizers (like TurboQuant)
that use geometric rotations instead of full orthogonal matrices.

Key advantages over TurboQuant:
  - PlanarQuant: 256 FMAs vs 16,384 → 64x fewer ops, ~28% faster decode
  - IsoQuant: 512 FMAs vs 16,384 → 32x fewer ops, better PPL
  - Both: dramatically fewer parameters (2-4 per block vs d² rotation matrix)

Methods:
  planar3/planar4 — 2D Givens rotation, 3 or 4 bit (simplest, fastest)
  iso3/iso4       — 4D quaternion rotation, 3 or 4 bit (best quality/speed tradeoff)

Reference: scrya-com/rotorquant (ICLR 2026)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Lloyd-Max codebook (shared with TurboQuant — same N(0,1) optimal centroids)
# ---------------------------------------------------------------------------

_CODEBOOK_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def _compute_lloyd_max_gaussian(
    n_levels: int, n_iters: int = 200
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Compute Lloyd-Max optimal centroids and boundaries for N(0,1)."""
    import numpy as np
    from scipy.stats import norm

    boundaries = np.linspace(-3.5, 3.5, n_levels + 1)
    boundaries[0] = -1e10
    boundaries[-1] = 1e10
    centroids = np.zeros(n_levels)

    for _ in range(n_iters):
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            p = norm.cdf(hi) - norm.cdf(lo)
            if p > 1e-15:
                centroids[i] = (norm.pdf(lo) - norm.pdf(hi)) / p
            else:
                centroids[i] = (max(lo, -3.5) + min(hi, 3.5)) / 2
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    return centroids, boundaries


def get_codebook(bit_width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get (centroids, inner_boundaries) for given bit-width, cached globally."""
    if bit_width not in _CODEBOOK_CACHE:
        import numpy as np
        n_levels = 2**bit_width
        centroids, boundaries = _compute_lloyd_max_gaussian(n_levels)
        _CODEBOOK_CACHE[bit_width] = (
            torch.tensor(centroids, dtype=torch.float32),
            torch.tensor(boundaries[1:-1], dtype=torch.float32),
        )
    return _CODEBOOK_CACHE[bit_width]


# ---------------------------------------------------------------------------
# PlanarQuant: 2D Givens rotation
# ---------------------------------------------------------------------------


def generate_givens_rotations(
    head_dim: int, seed: int = 42
) -> torch.Tensor:
    """Generate random Givens rotation angles for PlanarQuant.

    Returns rot2: (n_groups, 2) where each row is [cos(θ), sin(θ)].
    n_groups = head_dim // 2 (pairs of dimensions).
    """
    n_groups = head_dim // 2
    gen = torch.Generator().manual_seed(seed)
    angles = torch.rand(n_groups, generator=gen) * 2 * math.pi
    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)
    return torch.stack([cos_t, sin_t], dim=-1)  # (n_groups, 2)


def planar_quantize(
    x: torch.Tensor,
    rot2: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PlanarQuant: normalize → Givens rotate → scalar quantize.

    Args:
        x: (..., head_dim) input vectors
        rot2: (n_groups, 2) Givens parameters [cos, sin]
        centroids: (n_levels,) Lloyd-Max centroids
        boundaries: (n_levels-1,) Lloyd-Max boundaries

    Returns:
        indices: (..., head_dim) int8 quantized indices
        norms: (...,) FP32 vector norms
    """
    shape = x.shape
    head_dim = shape[-1]
    n_groups = head_dim // 2
    flat = x.reshape(-1, head_dim).float()

    norms = flat.norm(dim=-1).clamp(min=1e-8)
    flat_norm = flat / norms.unsqueeze(-1)

    cos_t = rot2[:, 0]  # (n_groups,)
    sin_t = rot2[:, 1]

    v0 = flat_norm[:, 0::2]  # (..., n_groups)
    v1 = flat_norm[:, 1::2]

    r0 = cos_t * v0 - sin_t * v1
    r1 = sin_t * v0 + cos_t * v1

    # Interleave back for searchsorted
    rotated = torch.empty_like(flat_norm)
    rotated[:, 0::2] = r0
    rotated[:, 1::2] = r1

    # Scalar quantize via searchsorted
    indices = torch.searchsorted(boundaries, rotated.reshape(-1))
    n_levels = centroids.shape[0]
    indices = indices.clamp(0, n_levels - 1).reshape_as(rotated).to(torch.int8)

    norms_out = norms.reshape(shape[:-1])
    indices_out = indices.reshape(shape)
    return indices_out, norms_out


def planar_dequantize(
    indices: torch.Tensor,
    norms: torch.Tensor,
    rot2: torch.Tensor,
    centroids: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """PlanarQuant: lookup centroids → inverse Givens rotate → rescale.

    Args:
        indices: (..., head_dim) int8 quantized indices
        norms: (...,) FP16/FP32 norms
        rot2: (n_groups, 2) Givens parameters [cos, sin]
        centroids: (n_levels,) Lloyd-Max centroids
        head_dim: original dimension

    Returns:
        x_hat: (..., head_dim) reconstructed vectors
    """
    shape = indices.shape
    flat_idx = indices.reshape(-1, head_dim).long()

    values = centroids[flat_idx]  # (..., head_dim) float32

    cos_t = rot2[:, 0]
    sin_t = rot2[:, 1]

    q0 = values[:, 0::2]
    q1 = values[:, 1::2]

    # Inverse Givens: R^T = [[cos, sin], [-sin, cos]]
    f0 = cos_t * q0 + sin_t * q1
    f1 = -sin_t * q0 + cos_t * q1

    result = torch.empty_like(values)
    result[:, 0::2] = f0
    result[:, 1::2] = f1

    norms_flat = norms.reshape(-1).float().unsqueeze(-1)
    result = result * norms_flat

    return result.reshape(shape).to(torch.float16)


# ---------------------------------------------------------------------------
# IsoQuant: 4D quaternion rotation
# ---------------------------------------------------------------------------


def generate_quaternion_rotations(
    head_dim: int, seed: int = 42
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Generate random unit quaternions for IsoQuant.

    Returns:
        q_L: (n_groups, 4) left quaternions
        q_R: (n_groups, 4) right quaternions (for full SO(4) mode)
    """
    n_groups = (head_dim + 3) // 4
    gen = torch.Generator().manual_seed(seed)
    q_L = torch.randn(n_groups, 4, generator=gen)
    q_L = q_L / q_L.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    gen2 = torch.Generator().manual_seed(seed + 10000)
    q_R = torch.randn(n_groups, 4, generator=gen2)
    q_R = q_R / q_R.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return q_L, q_R


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product: a * b for quaternions [w, x, y, z]."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([rw, rx, ry, rz], dim=-1)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate: (w, x, y, z) → (w, -x, -y, -z)."""
    signs = torch.tensor([1, -1, -1, -1], dtype=q.dtype, device=q.device)
    return q * signs


def iso_quantize(
    x: torch.Tensor,
    q_L: torch.Tensor,
    q_R: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IsoQuant: normalize → embed 4D → quaternion sandwich → scalar quantize.

    Args:
        x: (..., head_dim) input vectors
        q_L: (n_groups, 4) left quaternions
        q_R: (n_groups, 4) right quaternions
        centroids: (n_levels,) Lloyd-Max centroids
        boundaries: (n_levels-1,) Lloyd-Max boundaries

    Returns:
        indices: (..., d_padded) int8 quantized indices
        norms: (...,) FP32 vector norms
    """
    shape = x.shape
    head_dim = shape[-1]
    n_groups = (head_dim + 3) // 4
    d_padded = n_groups * 4

    flat = x.reshape(-1, head_dim).float()
    norms = flat.norm(dim=-1).clamp(min=1e-8)
    flat_norm = flat / norms.unsqueeze(-1)

    # Pad to multiple of 4
    if d_padded > head_dim:
        flat_norm = torch.nn.functional.pad(flat_norm, (0, d_padded - head_dim))

    # Reshape into 4D blocks: (batch, n_groups, 4)
    v = flat_norm.reshape(-1, n_groups, 4)

    # Quaternion sandwich: q_L * v * conj(q_R)
    temp = _quat_multiply(q_L, v)
    v_rot = _quat_multiply(temp, _quat_conjugate(q_R))

    # Scalar quantize
    rotated_flat = v_rot.reshape(-1, d_padded)
    indices = torch.searchsorted(boundaries, rotated_flat.reshape(-1))
    n_levels = centroids.shape[0]
    indices = indices.clamp(0, n_levels - 1).reshape(-1, d_padded).to(torch.int8)

    norms_out = norms.reshape(shape[:-1])
    indices_out = indices.reshape(*shape[:-1], d_padded)
    return indices_out, norms_out


def iso_dequantize(
    indices: torch.Tensor,
    norms: torch.Tensor,
    q_L: torch.Tensor,
    q_R: torch.Tensor,
    centroids: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """IsoQuant: lookup centroids → inverse quaternion sandwich → rescale.

    Args:
        indices: (..., d_padded) int8 quantized indices
        norms: (...,) norms
        q_L, q_R: quaternion rotations
        centroids: Lloyd-Max centroids
        head_dim: original dimension

    Returns:
        x_hat: (..., head_dim) reconstructed vectors
    """
    shape = indices.shape
    d_padded = shape[-1]
    n_groups = d_padded // 4

    flat_idx = indices.reshape(-1, d_padded).long()
    values = centroids[flat_idx]  # (batch, d_padded)

    v_q = values.reshape(-1, n_groups, 4)

    # Inverse sandwich: conj(q_L) * v * q_R
    temp = _quat_multiply(_quat_conjugate(q_L), v_q)
    v_recon = _quat_multiply(temp, q_R)

    result = v_recon.reshape(-1, d_padded)[:, :head_dim]

    norms_flat = norms.reshape(-1).float().unsqueeze(-1)
    result = result * norms_flat

    return result.reshape(*shape[:-1], head_dim).to(torch.float16)


# ---------------------------------------------------------------------------
# Bit packing (reuse same format as TurboQuant for compatibility)
# ---------------------------------------------------------------------------


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices (0-15) into uint8, 2 per byte."""
    assert indices.shape[-1] % 2 == 0
    lo = indices[..., 0::2].to(torch.uint8)
    hi = indices[..., 1::2].to(torch.uint8)
    return lo | (hi << 4)


def unpack_4bit(packed: torch.Tensor, orig_last_dim: int) -> torch.Tensor:
    """Unpack uint8 → 4-bit indices as int32."""
    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)
    result = torch.stack([lo, hi], dim=-1)
    return result.reshape(*packed.shape[:-1], packed.shape[-1] * 2)[..., :orig_last_dim]


def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit indices (0-7) into uint8. 8 values → 3 bytes."""
    d = indices.shape[-1]
    assert d % 8 == 0, f"Last dim must be multiple of 8, got {d}"
    idx = indices.to(torch.uint8)
    shape = idx.shape[:-1]
    idx = idx.reshape(*shape, d // 8, 8)
    v = [idx[..., i] for i in range(8)]
    b0 = v[0] | (v[1] << 3) | ((v[2] & 0x03) << 6)
    b1 = (v[2] >> 2) | (v[3] << 1) | (v[4] << 4) | ((v[5] & 0x01) << 7)
    b2 = (v[5] >> 1) | (v[6] << 2) | (v[7] << 5)
    return torch.stack([b0, b1, b2], dim=-1).reshape(*shape, d * 3 // 8)


def unpack_3bit(packed: torch.Tensor, orig_last_dim: int) -> torch.Tensor:
    """Unpack 3-bit packed bytes → int32 indices."""
    shape = packed.shape[:-1]
    n_groups = packed.shape[-1] // 3
    p = packed.reshape(*shape, n_groups, 3)
    b0 = p[..., 0].to(torch.int32)
    b1 = p[..., 1].to(torch.int32)
    b2 = p[..., 2].to(torch.int32)
    v0 = b0 & 0x07
    v1 = (b0 >> 3) & 0x07
    v2 = ((b0 >> 6) | (b1 << 2)) & 0x07
    v3 = (b1 >> 1) & 0x07
    v4 = (b1 >> 4) & 0x07
    v5 = ((b1 >> 7) | (b2 << 1)) & 0x07
    v6 = (b2 >> 2) & 0x07
    v7 = (b2 >> 5) & 0x07
    result = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)
    return result.reshape(*shape, n_groups * 8)[..., :orig_last_dim]


def pack_indices(indices: torch.Tensor, bit_width: int) -> torch.Tensor:
    if bit_width == 4:
        return pack_4bit(indices)
    elif bit_width == 3:
        return pack_3bit(indices)
    raise ValueError(f"Unsupported bit_width: {bit_width}")


def unpack_indices(packed: torch.Tensor, orig_last_dim: int, bit_width: int) -> torch.Tensor:
    if bit_width == 4:
        return unpack_4bit(packed, orig_last_dim)
    elif bit_width == 3:
        return unpack_3bit(packed, orig_last_dim)
    raise ValueError(f"Unsupported bit_width: {bit_width}")


def packed_bytes_per_dim(n_dims: int, bit_width: int) -> int:
    if bit_width == 4:
        return n_dims // 2
    elif bit_width == 3:
        return n_dims * 3 // 8
    raise ValueError(f"Unsupported bit_width: {bit_width}")


def pad_for_packing(n_dims: int, bit_width: int) -> int:
    if bit_width == 4:
        return n_dims + (n_dims % 2)
    elif bit_width == 3:
        return n_dims + (8 - n_dims % 8) % 8
    raise ValueError(f"Unsupported bit_width: {bit_width}")


# ---------------------------------------------------------------------------
# Triton PlanarQuant kernels (HIP/ROCm compatible via Triton compiler)
# ---------------------------------------------------------------------------

_triton_available = False
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _planar2_quantize_kernel(
        input_ptr, indices_ptr,
        rot2_ptr, centroids_ptr,
        batch_size, emb_dim,
        n_groups: tl.constexpr,
        n_levels: tl.constexpr,
        stride_in_b, stride_in_d,
        stride_idx_b, stride_idx_d,
        BLOCK_G: tl.constexpr,
    ):
        """Triton: Givens rotate → nearest centroid → store int8 index."""
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)

        g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        g_mask = g_offs < n_groups

        cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
        sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

        d0 = g_offs * 2
        v0 = tl.load(
            input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
            mask=g_mask & (d0 < emb_dim), other=0.0,
        )
        v1 = tl.load(
            input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
            mask=g_mask & ((d0 + 1) < emb_dim), other=0.0,
        )

        r0 = cos_t * v0 - sin_t * v1
        r1 = sin_t * v0 + cos_t * v1

        # Nearest centroid search for r0
        best_idx0 = tl.zeros_like(r0).to(tl.int32)
        best_dist0 = tl.abs(r0 - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            dd = tl.abs(r0 - c)
            mask = dd < best_dist0
            best_dist0 = tl.where(mask, dd, best_dist0)
            best_idx0 = tl.where(mask, i, best_idx0)

        # Nearest centroid search for r1
        best_idx1 = tl.zeros_like(r1).to(tl.int32)
        best_dist1 = tl.abs(r1 - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            dd = tl.abs(r1 - c)
            mask = dd < best_dist1
            best_dist1 = tl.where(mask, dd, best_dist1)
            best_idx1 = tl.where(mask, i, best_idx1)

        tl.store(
            indices_ptr + pid_b * stride_idx_b + d0 * stride_idx_d,
            best_idx0.to(tl.int8),
            mask=g_mask & (d0 < emb_dim),
        )
        tl.store(
            indices_ptr + pid_b * stride_idx_b + (d0 + 1) * stride_idx_d,
            best_idx1.to(tl.int8),
            mask=g_mask & ((d0 + 1) < emb_dim),
        )

    @triton.jit
    def _planar2_dequantize_kernel(
        indices_ptr, output_ptr,
        rot2_ptr, centroids_ptr,
        batch_size, emb_dim,
        n_groups: tl.constexpr,
        n_levels: tl.constexpr,
        stride_idx_b, stride_idx_d,
        stride_out_b, stride_out_d,
        BLOCK_G: tl.constexpr,
    ):
        """Triton: lookup centroids → inverse Givens rotate."""
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)

        g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        g_mask = g_offs < n_groups

        cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
        sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

        d0 = g_offs * 2

        idx0 = tl.load(
            indices_ptr + pid_b * stride_idx_b + d0 * stride_idx_d,
            mask=g_mask & (d0 < emb_dim), other=0,
        ).to(tl.int32)
        idx1 = tl.load(
            indices_ptr + pid_b * stride_idx_b + (d0 + 1) * stride_idx_d,
            mask=g_mask & ((d0 + 1) < emb_dim), other=0,
        ).to(tl.int32)

        q0 = tl.load(centroids_ptr + idx0, mask=g_mask, other=0.0)
        q1 = tl.load(centroids_ptr + idx1, mask=g_mask, other=0.0)

        # Inverse Givens: R^T
        f0 = cos_t * q0 + sin_t * q1
        f1 = -sin_t * q0 + cos_t * q1

        tl.store(
            output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
            f0, mask=g_mask & (d0 < emb_dim),
        )
        tl.store(
            output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
            f1, mask=g_mask & ((d0 + 1) < emb_dim),
        )

    # -------------------------------------------------------------------
    # Fused PlanarQuant 4-bit: rotate → quantize → pack in one kernel
    # -------------------------------------------------------------------

    @triton.jit
    def _fused_planar4_quant_pack_kernel(
        input_ptr, packed_ptr,
        rot2_ptr, centroids_ptr,
        batch_size, emb_dim,
        n_groups: tl.constexpr,
        n_levels: tl.constexpr,
        stride_in_b, stride_in_d,
        stride_pack_b, stride_pack_g,
        BLOCK_G: tl.constexpr,
    ):
        """Fused: Givens rotate → nearest centroid → 4-bit pack.
        Each Givens group (2 dims) → 1 packed byte (lo|hi<<4)."""
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)

        g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        g_mask = g_offs < n_groups

        cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
        sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

        d0 = g_offs * 2
        v0 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
                      mask=g_mask & (d0 < emb_dim), other=0.0)
        v1 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
                      mask=g_mask & ((d0 + 1) < emb_dim), other=0.0)

        r0 = cos_t * v0 - sin_t * v1
        r1 = sin_t * v0 + cos_t * v1

        best_idx0 = tl.zeros_like(r0).to(tl.int32)
        best_dist0 = tl.abs(r0 - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            dd = tl.abs(r0 - c)
            mask = dd < best_dist0
            best_dist0 = tl.where(mask, dd, best_dist0)
            best_idx0 = tl.where(mask, i, best_idx0)

        best_idx1 = tl.zeros_like(r1).to(tl.int32)
        best_dist1 = tl.abs(r1 - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            dd = tl.abs(r1 - c)
            mask = dd < best_dist1
            best_dist1 = tl.where(mask, dd, best_dist1)
            best_idx1 = tl.where(mask, i, best_idx1)

        packed_byte = (best_idx0 & 0x0F) | ((best_idx1 & 0x0F) << 4)

        tl.store(packed_ptr + pid_b * stride_pack_b + g_offs * stride_pack_g,
                 packed_byte.to(tl.int8), mask=g_mask)

    # -------------------------------------------------------------------
    # Fused PlanarQuant 4-bit: unpack → dequant → inverse rotate → rescale
    # -------------------------------------------------------------------

    @triton.jit
    def _fused_planar4_unpack_dequant_kernel(
        packed_ptr, output_ptr,
        rot2_ptr, centroids_ptr, norms_ptr,
        batch_size, emb_dim,
        n_groups: tl.constexpr,
        stride_pack_b, stride_pack_g,
        stride_out_b, stride_out_d,
        BLOCK_G: tl.constexpr,
    ):
        """Fused: unpack 4-bit → centroid lookup → inv Givens → rescale → fp16."""
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)

        g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        g_mask = g_offs < n_groups

        packed = tl.load(packed_ptr + pid_b * stride_pack_b + g_offs * stride_pack_g,
                         mask=g_mask, other=0).to(tl.int32)
        idx0 = packed & 0x0F
        idx1 = (packed >> 4) & 0x0F

        q0 = tl.load(centroids_ptr + idx0, mask=g_mask, other=0.0)
        q1 = tl.load(centroids_ptr + idx1, mask=g_mask, other=0.0)

        cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
        sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

        f0 = cos_t * q0 + sin_t * q1
        f1 = -sin_t * q0 + cos_t * q1

        norm = tl.load(norms_ptr + pid_b).to(tl.float32)
        f0 = f0 * norm
        f1 = f1 * norm

        d0 = g_offs * 2
        tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
                 f0.to(tl.float16), mask=g_mask & (d0 < emb_dim))
        tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
                 f1.to(tl.float16), mask=g_mask & ((d0 + 1) < emb_dim))

    # -------------------------------------------------------------------
    # IsoQuant: quaternion sandwich → quantize (non-fused, for 3-bit path)
    # -------------------------------------------------------------------

    @triton.jit
    def _iso_quantize_kernel(
        input_ptr, indices_ptr,
        qL_ptr, qR_ptr, centroids_ptr,
        batch_size, d_padded,
        n_groups: tl.constexpr,
        n_levels: tl.constexpr,
        stride_in_b, stride_in_d,
        stride_idx_b, stride_idx_d,
        BLOCK_G: tl.constexpr,
    ):
        """Quaternion sandwich q_L * v * conj(q_R) → nearest centroid."""
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)

        g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        g_mask = g_offs < n_groups

        d0 = g_offs * 4
        v0 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
                      mask=g_mask, other=0.0)
        v1 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
                      mask=g_mask, other=0.0)
        v2 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 2) * stride_in_d,
                      mask=g_mask, other=0.0)
        v3 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 3) * stride_in_d,
                      mask=g_mask, other=0.0)

        aw = tl.load(qL_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
        ax = tl.load(qL_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
        ay = tl.load(qL_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
        az = tl.load(qL_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

        bw = tl.load(qR_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
        bx = tl.load(qR_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
        by = tl.load(qR_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
        bz = tl.load(qR_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

        # Hamilton product: temp = q_L * v
        tw = aw * v0 - ax * v1 - ay * v2 - az * v3
        tx = aw * v1 + ax * v0 + ay * v3 - az * v2
        ty = aw * v2 - ax * v3 + ay * v0 + az * v1
        tz = aw * v3 + ax * v2 - ay * v1 + az * v0

        # Hamilton product: result = temp * conj(q_R)
        rw = tw * bw + tx * bx + ty * by + tz * bz
        rx = -tw * bx + tx * bw - ty * bz + tz * by
        ry = -tw * by + tx * bz + ty * bw - tz * bx
        rz = -tw * bz - tx * by + ty * bx + tz * bw

        # Nearest centroid for each of the 4 components
        best_iw = tl.zeros_like(rw).to(tl.int32)
        best_dw = tl.abs(rw - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            dd = tl.abs(rw - c)
            m = dd < best_dw
            best_dw = tl.where(m, dd, best_dw)
            best_iw = tl.where(m, i, best_iw)

        best_ix = tl.zeros_like(rx).to(tl.int32)
        best_dx = tl.abs(rx - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            dd = tl.abs(rx - c)
            m = dd < best_dx
            best_dx = tl.where(m, dd, best_dx)
            best_ix = tl.where(m, i, best_ix)

        best_iy = tl.zeros_like(ry).to(tl.int32)
        best_dy = tl.abs(ry - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            dd = tl.abs(ry - c)
            m = dd < best_dy
            best_dy = tl.where(m, dd, best_dy)
            best_iy = tl.where(m, i, best_iy)

        best_iz = tl.zeros_like(rz).to(tl.int32)
        best_dz = tl.abs(rz - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            dd = tl.abs(rz - c)
            m = dd < best_dz
            best_dz = tl.where(m, dd, best_dz)
            best_iz = tl.where(m, i, best_iz)

        tl.store(indices_ptr + pid_b * stride_idx_b + d0 * stride_idx_d,
                 best_iw.to(tl.int8), mask=g_mask)
        tl.store(indices_ptr + pid_b * stride_idx_b + (d0 + 1) * stride_idx_d,
                 best_ix.to(tl.int8), mask=g_mask)
        tl.store(indices_ptr + pid_b * stride_idx_b + (d0 + 2) * stride_idx_d,
                 best_iy.to(tl.int8), mask=g_mask)
        tl.store(indices_ptr + pid_b * stride_idx_b + (d0 + 3) * stride_idx_d,
                 best_iz.to(tl.int8), mask=g_mask)

    # -------------------------------------------------------------------
    # IsoQuant: dequantize → inverse quaternion sandwich (non-fused)
    # -------------------------------------------------------------------

    @triton.jit
    def _iso_dequantize_kernel(
        indices_ptr, output_ptr,
        qL_ptr, qR_ptr, centroids_ptr,
        batch_size, d_padded,
        n_groups: tl.constexpr,
        stride_idx_b, stride_idx_d,
        stride_out_b, stride_out_d,
        BLOCK_G: tl.constexpr,
    ):
        """Centroid lookup → conj(q_L) * v * q_R (inverse sandwich)."""
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)

        g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        g_mask = g_offs < n_groups

        d0 = g_offs * 4
        iw = tl.load(indices_ptr + pid_b * stride_idx_b + d0 * stride_idx_d,
                      mask=g_mask, other=0).to(tl.int32)
        ix = tl.load(indices_ptr + pid_b * stride_idx_b + (d0 + 1) * stride_idx_d,
                      mask=g_mask, other=0).to(tl.int32)
        iy = tl.load(indices_ptr + pid_b * stride_idx_b + (d0 + 2) * stride_idx_d,
                      mask=g_mask, other=0).to(tl.int32)
        iz = tl.load(indices_ptr + pid_b * stride_idx_b + (d0 + 3) * stride_idx_d,
                      mask=g_mask, other=0).to(tl.int32)

        v0 = tl.load(centroids_ptr + iw, mask=g_mask, other=0.0)
        v1 = tl.load(centroids_ptr + ix, mask=g_mask, other=0.0)
        v2 = tl.load(centroids_ptr + iy, mask=g_mask, other=0.0)
        v3 = tl.load(centroids_ptr + iz, mask=g_mask, other=0.0)

        aw = tl.load(qL_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
        ax = tl.load(qL_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
        ay = tl.load(qL_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
        az = tl.load(qL_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

        bw = tl.load(qR_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
        bx = tl.load(qR_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
        by = tl.load(qR_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
        bz = tl.load(qR_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

        # Hamilton product: temp = conj(q_L) * v
        tw = aw * v0 + ax * v1 + ay * v2 + az * v3
        tx = aw * v1 - ax * v0 - ay * v3 + az * v2
        ty = aw * v2 + ax * v3 - ay * v0 - az * v1
        tz = aw * v3 - ax * v2 + ay * v1 - az * v0

        # Hamilton product: result = temp * q_R
        rw = tw * bw - tx * bx - ty * by - tz * bz
        rx = tw * bx + tx * bw + ty * bz - tz * by
        ry = tw * by - tx * bz + ty * bw + tz * bx
        rz = tw * bz + tx * by - ty * bx + tz * bw

        tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
                 rw, mask=g_mask)
        tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
                 rx, mask=g_mask)
        tl.store(output_ptr + pid_b * stride_out_b + (d0 + 2) * stride_out_d,
                 ry, mask=g_mask)
        tl.store(output_ptr + pid_b * stride_out_b + (d0 + 3) * stride_out_d,
                 rz, mask=g_mask)

    # -------------------------------------------------------------------
    # Fused IsoQuant 4-bit: quat sandwich → quantize → pack 2 bytes/group
    # -------------------------------------------------------------------

    @triton.jit
    def _fused_iso4_quant_pack_kernel(
        input_ptr, packed_ptr,
        qL_ptr, qR_ptr, centroids_ptr,
        batch_size, d_padded,
        n_groups: tl.constexpr,
        n_levels: tl.constexpr,
        stride_in_b, stride_in_d,
        stride_pack_b, stride_pack_e,
        BLOCK_G: tl.constexpr,
    ):
        """Fused: quat sandwich → nearest centroid → 4-bit pack (2 bytes/group)."""
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)

        g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        g_mask = g_offs < n_groups

        d0 = g_offs * 4
        v0 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
                      mask=g_mask, other=0.0)
        v1 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
                      mask=g_mask, other=0.0)
        v2 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 2) * stride_in_d,
                      mask=g_mask, other=0.0)
        v3 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 3) * stride_in_d,
                      mask=g_mask, other=0.0)

        aw = tl.load(qL_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
        ax = tl.load(qL_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
        ay = tl.load(qL_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
        az = tl.load(qL_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

        bw = tl.load(qR_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
        bx = tl.load(qR_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
        by = tl.load(qR_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
        bz = tl.load(qR_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

        tw = aw * v0 - ax * v1 - ay * v2 - az * v3
        tx = aw * v1 + ax * v0 + ay * v3 - az * v2
        ty = aw * v2 - ax * v3 + ay * v0 + az * v1
        tz = aw * v3 + ax * v2 - ay * v1 + az * v0

        rw = tw * bw + tx * bx + ty * by + tz * bz
        rx = -tw * bx + tx * bw - ty * bz + tz * by
        ry = -tw * by + tx * bz + ty * bw - tz * bx
        rz = -tw * bz - tx * by + ty * bx + tz * bw

        # Nearest centroid × 4 components
        best_iw = tl.zeros_like(rw).to(tl.int32)
        best_dw = tl.abs(rw - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            m = tl.abs(rw - c) < best_dw
            best_dw = tl.where(m, tl.abs(rw - c), best_dw)
            best_iw = tl.where(m, i, best_iw)

        best_ix = tl.zeros_like(rx).to(tl.int32)
        best_dx = tl.abs(rx - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            m = tl.abs(rx - c) < best_dx
            best_dx = tl.where(m, tl.abs(rx - c), best_dx)
            best_ix = tl.where(m, i, best_ix)

        best_iy = tl.zeros_like(ry).to(tl.int32)
        best_dy = tl.abs(ry - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            m = tl.abs(ry - c) < best_dy
            best_dy = tl.where(m, tl.abs(ry - c), best_dy)
            best_iy = tl.where(m, i, best_iy)

        best_iz = tl.zeros_like(rz).to(tl.int32)
        best_dz = tl.abs(rz - tl.load(centroids_ptr))
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            m = tl.abs(rz - c) < best_dz
            best_dz = tl.where(m, tl.abs(rz - c), best_dz)
            best_iz = tl.where(m, i, best_iz)

        # Pack: 4 indices → 2 bytes (byte0 = iw|ix<<4, byte1 = iy|iz<<4)
        byte0 = (best_iw & 0x0F) | ((best_ix & 0x0F) << 4)
        byte1 = (best_iy & 0x0F) | ((best_iz & 0x0F) << 4)

        tl.store(packed_ptr + pid_b * stride_pack_b + (g_offs * 2) * stride_pack_e,
                 byte0.to(tl.int8), mask=g_mask)
        tl.store(packed_ptr + pid_b * stride_pack_b + (g_offs * 2 + 1) * stride_pack_e,
                 byte1.to(tl.int8), mask=g_mask)

    # -------------------------------------------------------------------
    # Fused IsoQuant 4-bit: unpack → dequant → inv quat sandwich → rescale
    # -------------------------------------------------------------------

    @triton.jit
    def _fused_iso4_unpack_dequant_kernel(
        packed_ptr, output_ptr,
        qL_ptr, qR_ptr, centroids_ptr, norms_ptr,
        batch_size, d_padded, head_dim,
        n_groups: tl.constexpr,
        stride_pack_b, stride_pack_e,
        stride_out_b, stride_out_d,
        BLOCK_G: tl.constexpr,
    ):
        """Fused: unpack 4-bit → centroid → inv quat sandwich → rescale → fp16."""
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)

        g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        g_mask = g_offs < n_groups

        byte0 = tl.load(packed_ptr + pid_b * stride_pack_b + (g_offs * 2) * stride_pack_e,
                         mask=g_mask, other=0).to(tl.int32)
        byte1 = tl.load(packed_ptr + pid_b * stride_pack_b + (g_offs * 2 + 1) * stride_pack_e,
                         mask=g_mask, other=0).to(tl.int32)

        iw = byte0 & 0x0F
        ix = (byte0 >> 4) & 0x0F
        iy = byte1 & 0x0F
        iz = (byte1 >> 4) & 0x0F

        v0 = tl.load(centroids_ptr + iw, mask=g_mask, other=0.0)
        v1 = tl.load(centroids_ptr + ix, mask=g_mask, other=0.0)
        v2 = tl.load(centroids_ptr + iy, mask=g_mask, other=0.0)
        v3 = tl.load(centroids_ptr + iz, mask=g_mask, other=0.0)

        aw = tl.load(qL_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
        ax = tl.load(qL_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
        ay = tl.load(qL_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
        az = tl.load(qL_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

        bw = tl.load(qR_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
        bx = tl.load(qR_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
        by = tl.load(qR_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
        bz = tl.load(qR_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

        # Inverse sandwich: conj(q_L) * v * q_R
        tw = aw * v0 + ax * v1 + ay * v2 + az * v3
        tx = aw * v1 - ax * v0 - ay * v3 + az * v2
        ty = aw * v2 + ax * v3 - ay * v0 - az * v1
        tz = aw * v3 - ax * v2 + ay * v1 - az * v0

        rw = tw * bw - tx * bx - ty * by - tz * bz
        rx = tw * bx + tx * bw + ty * bz - tz * by
        ry = tw * by - tx * bz + ty * bw + tz * bx
        rz = tw * bz + tx * by - ty * bx + tz * bw

        norm = tl.load(norms_ptr + pid_b).to(tl.float32)
        rw = rw * norm
        rx = rx * norm
        ry = ry * norm
        rz = rz * norm

        d0 = g_offs * 4
        tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
                 rw.to(tl.float16), mask=g_mask & (d0 < head_dim))
        tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
                 rx.to(tl.float16), mask=g_mask & ((d0 + 1) < head_dim))
        tl.store(output_ptr + pid_b * stride_out_b + (d0 + 2) * stride_out_d,
                 ry.to(tl.float16), mask=g_mask & ((d0 + 2) < head_dim))
        tl.store(output_ptr + pid_b * stride_out_b + (d0 + 3) * stride_out_d,
                 rz.to(tl.float16), mask=g_mask & ((d0 + 3) < head_dim))

    _triton_available = True
except ImportError:
    _triton_available = False


def triton_planar_quantize(
    x: torch.Tensor,
    rot2: torch.Tensor,
    centroids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated PlanarQuant quantize. Returns (int8 indices, norms)."""
    batch_size, emb_dim = x.shape
    n_groups = rot2.shape[0]
    n_levels = centroids.shape[0]

    x_f32 = x.float()
    norms = x_f32.norm(dim=-1).clamp(min=1e-8)
    x_f32 = (x_f32 / norms.unsqueeze(-1)).contiguous()

    rot2_f32 = rot2.float().contiguous()
    c_f32 = centroids.float().contiguous()

    indices = torch.empty(batch_size, emb_dim, dtype=torch.int8, device=x.device)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 256)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _planar2_quantize_kernel[grid](
        x_f32, indices, rot2_f32, c_f32,
        batch_size, emb_dim, n_groups, n_levels,
        x_f32.stride(0), x_f32.stride(1),
        indices.stride(0), indices.stride(1),
        BLOCK_G=BLOCK_G,
    )
    return indices, norms


def triton_planar_dequantize(
    indices: torch.Tensor,
    norms: torch.Tensor,
    rot2: torch.Tensor,
    centroids: torch.Tensor,
    emb_dim: int,
) -> torch.Tensor:
    """Triton-accelerated PlanarQuant dequantize. Returns FP16 vectors."""
    batch_size = indices.shape[0]
    n_groups = rot2.shape[0]
    n_levels = centroids.shape[0]

    rot2_f32 = rot2.float().contiguous()
    c_f32 = centroids.float().contiguous()

    output = torch.empty(batch_size, emb_dim, dtype=torch.float32, device=indices.device)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 256)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _planar2_dequantize_kernel[grid](
        indices.contiguous(), output, rot2_f32, c_f32,
        batch_size, emb_dim, n_groups, n_levels,
        indices.stride(0), indices.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    output = output * norms.float().unsqueeze(-1)
    return output.to(torch.float16)


def triton_fused_planar4_quant_pack(
    x: torch.Tensor,
    rot2: torch.Tensor,
    centroids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton PlanarQuant 4-bit: rotate+quantize+pack → (uint8 packed, norms)."""
    batch_size, emb_dim = x.shape
    n_groups = emb_dim // 2
    n_levels = centroids.shape[0]

    x_f32 = x.float()
    norms = x_f32.norm(dim=-1).clamp(min=1e-8)
    x_f32 = (x_f32 / norms.unsqueeze(-1)).contiguous()

    rot2_f32 = rot2.float().contiguous()
    c_f32 = centroids.float().contiguous()

    packed = torch.empty(batch_size, n_groups, dtype=torch.uint8, device=x.device)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 256)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _fused_planar4_quant_pack_kernel[grid](
        x_f32, packed, rot2_f32, c_f32,
        batch_size, emb_dim, n_groups, n_levels,
        x_f32.stride(0), x_f32.stride(1),
        packed.stride(0), packed.stride(1),
        BLOCK_G=BLOCK_G,
    )
    return packed, norms


def triton_fused_planar4_unpack_dequant(
    packed: torch.Tensor,
    norms: torch.Tensor,
    rot2: torch.Tensor,
    centroids: torch.Tensor,
    emb_dim: int,
) -> torch.Tensor:
    """Fused Triton PlanarQuant 4-bit: unpack+dequant+rescale → FP16."""
    batch_size = packed.shape[0]
    n_groups = emb_dim // 2

    rot2_f32 = rot2.float().contiguous()
    c_f32 = centroids.float().contiguous()
    norms_f32 = norms.float().contiguous()

    output = torch.empty(batch_size, emb_dim, dtype=torch.float16, device=packed.device)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 256)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _fused_planar4_unpack_dequant_kernel[grid](
        packed.contiguous(), output, rot2_f32, c_f32, norms_f32,
        batch_size, emb_dim, n_groups,
        packed.stride(0), packed.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_G=BLOCK_G,
    )
    return output


def triton_iso_quantize(
    x: torch.Tensor,
    q_L: torch.Tensor,
    q_R: torch.Tensor,
    centroids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton IsoQuant: quaternion sandwich → quantize. Returns (int8 indices, norms)."""
    shape = x.shape
    head_dim = shape[-1]
    n_groups = (head_dim + 3) // 4
    d_padded = n_groups * 4
    n_levels = centroids.shape[0]

    flat = x.reshape(-1, head_dim).float()
    batch_size = flat.shape[0]
    norms = flat.norm(dim=-1).clamp(min=1e-8)
    flat_norm = flat / norms.unsqueeze(-1)

    if d_padded > head_dim:
        flat_norm = torch.nn.functional.pad(flat_norm, (0, d_padded - head_dim))
    flat_norm = flat_norm.contiguous()

    qL_f32 = q_L.float().contiguous()
    qR_f32 = q_R.float().contiguous()
    c_f32 = centroids.float().contiguous()

    indices = torch.empty(batch_size, d_padded, dtype=torch.int8, device=x.device)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 128)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _iso_quantize_kernel[grid](
        flat_norm, indices, qL_f32, qR_f32, c_f32,
        batch_size, d_padded, n_groups, n_levels,
        flat_norm.stride(0), flat_norm.stride(1),
        indices.stride(0), indices.stride(1),
        BLOCK_G=BLOCK_G,
    )

    norms_out = norms.reshape(shape[:-1])
    indices_out = indices.reshape(*shape[:-1], d_padded)
    return indices_out, norms_out


def triton_iso_dequantize(
    indices: torch.Tensor,
    norms: torch.Tensor,
    q_L: torch.Tensor,
    q_R: torch.Tensor,
    centroids: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Triton IsoQuant dequantize: inv quaternion sandwich → rescale → FP16."""
    shape = indices.shape
    d_padded = shape[-1]
    n_groups = d_padded // 4

    flat_idx = indices.reshape(-1, d_padded).contiguous()
    batch_size = flat_idx.shape[0]

    qL_f32 = q_L.float().contiguous()
    qR_f32 = q_R.float().contiguous()
    c_f32 = centroids.float().contiguous()

    output = torch.empty(batch_size, d_padded, dtype=torch.float32, device=indices.device)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 128)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _iso_dequantize_kernel[grid](
        flat_idx, output, qL_f32, qR_f32, c_f32,
        batch_size, d_padded, n_groups,
        flat_idx.stride(0), flat_idx.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    result = output[:, :head_dim]
    norms_flat = norms.reshape(-1).float().unsqueeze(-1)
    result = result * norms_flat
    return result.reshape(*shape[:-1], head_dim).to(torch.float16)


def triton_fused_iso4_quant_pack(
    x: torch.Tensor,
    q_L: torch.Tensor,
    q_R: torch.Tensor,
    centroids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton IsoQuant 4-bit: quat sandwich+quantize+pack → (uint8, norms)."""
    shape = x.shape
    head_dim = shape[-1]
    n_groups = (head_dim + 3) // 4
    d_padded = n_groups * 4
    n_levels = centroids.shape[0]

    flat = x.reshape(-1, head_dim).float()
    batch_size = flat.shape[0]
    norms = flat.norm(dim=-1).clamp(min=1e-8)
    flat_norm = flat / norms.unsqueeze(-1)

    if d_padded > head_dim:
        flat_norm = torch.nn.functional.pad(flat_norm, (0, d_padded - head_dim))
    flat_norm = flat_norm.contiguous()

    qL_f32 = q_L.float().contiguous()
    qR_f32 = q_R.float().contiguous()
    c_f32 = centroids.float().contiguous()

    packed = torch.empty(batch_size, n_groups * 2, dtype=torch.uint8, device=x.device)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 128)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _fused_iso4_quant_pack_kernel[grid](
        flat_norm, packed, qL_f32, qR_f32, c_f32,
        batch_size, d_padded, n_groups, n_levels,
        flat_norm.stride(0), flat_norm.stride(1),
        packed.stride(0), packed.stride(1),
        BLOCK_G=BLOCK_G,
    )

    norms_out = norms.reshape(shape[:-1])
    packed_out = packed.reshape(*shape[:-1], n_groups * 2)
    return packed_out, norms_out


def triton_fused_iso4_unpack_dequant(
    packed: torch.Tensor,
    norms: torch.Tensor,
    q_L: torch.Tensor,
    q_R: torch.Tensor,
    centroids: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Fused Triton IsoQuant 4-bit: unpack+dequant+inv quat+rescale → FP16."""
    shape = packed.shape
    n_packed = shape[-1]
    n_groups = n_packed // 2
    d_padded = n_groups * 4

    flat_pack = packed.reshape(-1, n_packed).contiguous()
    batch_size = flat_pack.shape[0]

    qL_f32 = q_L.float().contiguous()
    qR_f32 = q_R.float().contiguous()
    c_f32 = centroids.float().contiguous()
    norms_f32 = norms.reshape(-1).float().contiguous()

    output = torch.empty(batch_size, head_dim, dtype=torch.float16, device=packed.device)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 128)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _fused_iso4_unpack_dequant_kernel[grid](
        flat_pack, output, qL_f32, qR_f32, c_f32, norms_f32,
        batch_size, d_padded, head_dim, n_groups,
        flat_pack.stride(0), flat_pack.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    return output.reshape(*shape[:-1], head_dim)


# ---------------------------------------------------------------------------
# Unified compress/decompress API for SGLang KV cache pool
# ---------------------------------------------------------------------------


def compress_kv_head(
    x: torch.Tensor,
    method: str,
    bit_width: int,
    rotations: dict,
    codebook: dict,
    device: torch.device,
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress a single KV head vector batch.

    Args:
        x: (T, head_dim) — T tokens, FP16/BF16
        method: 'planar' or 'iso'
        bit_width: 3 or 4
        rotations: dict with rotation parameters
        codebook: dict with 'centroids' and 'boundaries' tensors
        device: target device
        use_triton: try Triton kernels first

    Returns:
        packed: (T, packed_bytes) uint8 packed indices
        norms: (T,) FP16 norms
    """
    head_dim = x.shape[-1]
    centroids = codebook["centroids"]
    boundaries = codebook["boundaries"]

    if method == "planar":
        rot2 = rotations["rot2"]
        if use_triton and _triton_available and x.is_cuda and bit_width == 4:
            # Fused: rotate → quantize → pack in one kernel (no intermediate indices)
            packed, norms = triton_fused_planar4_quant_pack(x, rot2, centroids)
            return packed, norms.to(torch.float16)
        if use_triton and _triton_available and x.is_cuda:
            indices, norms = triton_planar_quantize(x, rot2, centroids)
        else:
            indices, norms = planar_quantize(x, rot2, centroids, boundaries)
        # indices shape: (T, head_dim) int8
        d_pack = pad_for_packing(head_dim, bit_width)
        if d_pack > head_dim:
            indices = torch.nn.functional.pad(indices.to(torch.int32), (0, d_pack - head_dim), value=0)
        else:
            indices = indices.to(torch.int32)
        packed = pack_indices(indices, bit_width)
    elif method == "iso":
        q_L = rotations["q_L"]
        q_R = rotations["q_R"]
        if use_triton and _triton_available and x.is_cuda and bit_width == 4:
            # Fused: quat sandwich → quantize → pack in one kernel
            packed, norms = triton_fused_iso4_quant_pack(x, q_L, q_R, centroids)
            return packed, norms.to(torch.float16)
        if use_triton and _triton_available and x.is_cuda:
            indices, norms = triton_iso_quantize(x, q_L, q_R, centroids)
        else:
            indices, norms = iso_quantize(x, q_L, q_R, centroids, boundaries)
        # indices shape: (T, d_padded) int8
        d_padded = indices.shape[-1]
        d_pack = pad_for_packing(d_padded, bit_width)
        if d_pack > d_padded:
            indices = torch.nn.functional.pad(indices.to(torch.int32), (0, d_pack - d_padded), value=0)
        else:
            indices = indices.to(torch.int32)
        packed = pack_indices(indices, bit_width)
    else:
        raise ValueError(f"Unknown RotorQuant method: {method}")

    return packed, norms.to(torch.float16)


def decompress_kv_head(
    packed: torch.Tensor,
    norms: torch.Tensor,
    method: str,
    bit_width: int,
    head_dim: int,
    rotations: dict,
    codebook: dict,
    device: torch.device,
    use_triton: bool = True,
) -> torch.Tensor:
    """Decompress a single KV head vector batch.

    Args:
        packed: (T, packed_bytes) uint8 packed indices
        norms: (T,) FP16 norms
        method: 'planar' or 'iso'
        bit_width: 3 or 4
        head_dim: original dimension
        rotations: dict with rotation parameters
        codebook: dict with 'centroids' tensor
        device: target device
        use_triton: try Triton kernels first

    Returns:
        x_hat: (T, head_dim) FP16 reconstructed vectors
    """
    centroids = codebook["centroids"]

    if method == "planar":
        rot2 = rotations["rot2"]
        if use_triton and _triton_available and packed.is_cuda and bit_width == 4:
            # Fused: unpack → dequant → inv rotate → rescale in one kernel
            x_hat = triton_fused_planar4_unpack_dequant(
                packed, norms, rot2, centroids, head_dim,
            )
        else:
            d_pack = pad_for_packing(head_dim, bit_width)
            indices = unpack_indices(packed, d_pack, bit_width)[:, :head_dim]
            if use_triton and _triton_available and packed.is_cuda:
                x_hat = triton_planar_dequantize(
                    indices.to(torch.int8).contiguous(),
                    norms, rot2, centroids, head_dim,
                )
            else:
                x_hat = planar_dequantize(indices.to(torch.int8), norms, rot2, centroids, head_dim)
    elif method == "iso":
        q_L = rotations["q_L"]
        q_R = rotations["q_R"]
        if use_triton and _triton_available and packed.is_cuda and bit_width == 4:
            # Fused: unpack → dequant → inv quat sandwich → rescale in one kernel
            x_hat = triton_fused_iso4_unpack_dequant(
                packed, norms, q_L, q_R, centroids, head_dim,
            )
        else:
            n_groups = (head_dim + 3) // 4
            d_padded = n_groups * 4
            d_pack = pad_for_packing(d_padded, bit_width)
            indices = unpack_indices(packed, d_pack, bit_width)[:, :d_padded]
            if use_triton and _triton_available and packed.is_cuda:
                x_hat = triton_iso_dequantize(
                    indices.to(torch.int8).contiguous(),
                    norms, q_L, q_R, centroids, head_dim,
                )
            else:
                x_hat = iso_dequantize(indices.to(torch.int8), norms, q_L, q_R, centroids, head_dim)
    else:
        raise ValueError(f"Unknown RotorQuant method: {method}")

    return x_hat
