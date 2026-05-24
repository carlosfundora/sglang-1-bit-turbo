"""Wave32-optimized fused QKNorm + RoPE HIP kernel for RDNA2 (gfx1030)."""

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

RDNA2_FUSED_QKNORM_ROPE_DECL = """
void rdna2_fused_qknorm_rope(
    torch::Tensor& qkv, torch::Tensor& q_weight, torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache, torch::Tensor& positions,
    int num_heads_q, int num_heads_k, int num_heads_v,
    int head_dim, int rotary_dim, float eps, bool is_neox);
"""

RDNA2_FUSED_QKNORM_ROPE_CU = r"""
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

using __nv_bfloat16 = __hip_bfloat16;

namespace rdna2 {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;  // 128

template <typename scalar_t, int HEAD_DIM, bool IS_NEOX>
__global__ __attribute__((amdgpu_flat_work_group_size(128, 128)))
void fused_qknorm_rope_kernel(
    scalar_t* __restrict__ qkv,
    const scalar_t* __restrict__ q_weight,
    const scalar_t* __restrict__ k_weight,
    const scalar_t* __restrict__ cos_sin_cache,
    const int64_t* __restrict__ positions,
    const int num_heads_q,
    const int num_heads_k,
    const int num_heads_v,
    const int rotary_dim,
    const float eps,
    const int num_tokens)
{
    constexpr int ELEMS_PER_THREAD = HEAD_DIM / WARP_SIZE;
    static_assert(HEAD_DIM % WARP_SIZE == 0,
                  "HEAD_DIM must be divisible by WARP_SIZE (32)");
    static_assert(ELEMS_PER_THREAD >= 2,
                  "HEAD_DIM must be >= 64 for fused QKNorm+RoPE");

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    const int total_qk = num_heads_q + num_heads_k;
    const int token_idx = global_warp / total_qk;
    const int local_head = global_warp % total_qk;

    if (token_idx >= num_tokens) return;

    const bool is_q = local_head < num_heads_q;
    const int head_idx = is_q ? local_head : (local_head - num_heads_q);
    const int total_heads = num_heads_q + num_heads_k + num_heads_v;

    int head_offset = is_q
        ? head_idx * HEAD_DIM
        : (num_heads_q + head_idx) * HEAD_DIM;
    int qkv_base = token_idx * total_heads * HEAD_DIM + head_offset;

    const scalar_t* weight = is_q ? q_weight : k_weight;

    float elems[ELEMS_PER_THREAD];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        float val = float(qkv[qkv_base + lane * ELEMS_PER_THREAD + i]);
        elems[i] = val;
        sum_sq += val * val;
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum_sq += __shfl_xor(sum_sq, mask);

    float rms_inv = rsqrtf(sum_sq / float(HEAD_DIM) + eps);

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int dim = lane * ELEMS_PER_THREAD + i;
        elems[i] = elems[i] * rms_inv * float(weight[dim]);
    }

    const int embed_dim = rotary_dim / 2;
    const int64_t pos = positions[token_idx];
    const scalar_t* cos_ptr = cos_sin_cache + pos * rotary_dim;
    const scalar_t* sin_ptr = cos_ptr + embed_dim;

    const int rotary_lanes = rotary_dim / ELEMS_PER_THREAD;

    if (lane < rotary_lanes) {
        if constexpr (IS_NEOX) {
            const int pair_offset = embed_dim / ELEMS_PER_THREAD;

            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i++) {
                float partner = __shfl_xor(elems[i], pair_offset);
                if (lane < pair_offset) partner = -partner;

                int dim = lane * ELEMS_PER_THREAD + i;
                int mapped_dim = (dim * 2) % rotary_dim;
                int half_dim = mapped_dim / 2;

                float cos_val = float(cos_ptr[half_dim]);
                float sin_val = float(sin_ptr[half_dim]);

                elems[i] = elems[i] * cos_val + partner * sin_val;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD / 2; i++) {
                int idx0 = 2 * i;
                int idx1 = 2 * i + 1;
                int dim = lane * ELEMS_PER_THREAD + idx0;
                int half_dim = dim / 2;

                float cos_val = float(cos_ptr[half_dim]);
                float sin_val = float(sin_ptr[half_dim]);

                float v0 = elems[idx0];
                float v1 = elems[idx1];
                elems[idx0] = v0 * cos_val - v1 * sin_val;
                elems[idx1] = v0 * sin_val + v1 * cos_val;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        qkv[qkv_base + lane * ELEMS_PER_THREAD + i] = scalar_t(elems[i]);
    }
}

}  // namespace rdna2

void rdna2_fused_qknorm_rope(
    torch::Tensor& qkv,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache,
    torch::Tensor& positions,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    int head_dim,
    int rotary_dim,
    float eps,
    bool is_neox)
{
    int num_tokens = positions.size(0);
    int total_qk = num_heads_q + num_heads_k;
    int total_heads = num_heads_q + num_heads_k + num_heads_v;
    int total_warps = num_tokens * total_qk;

    TORCH_CHECK(qkv.is_contiguous(), "qkv must be contiguous");
    TORCH_CHECK(cos_sin_cache.is_contiguous(), "cos_sin_cache must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even, got ", rotary_dim);
    TORCH_CHECK(rotary_dim <= head_dim, "rotary_dim (", rotary_dim,
                ") must be <= head_dim (", head_dim, ")");
    TORCH_CHECK(qkv.size(1) == total_heads * head_dim,
                "qkv dim1 (", qkv.size(1), ") != total_heads*head_dim (",
                total_heads * head_dim, ")");
    TORCH_CHECK(q_weight.size(0) == head_dim, "q_weight size mismatch");
    TORCH_CHECK(k_weight.size(0) == head_dim, "k_weight size mismatch");
    if (is_neox) {
        int embed_dim = rotary_dim / 2;
        int elems_per_thread = head_dim / rdna2::WARP_SIZE;
        TORCH_CHECK(embed_dim % elems_per_thread == 0,
                    "NeoX: embed_dim (", embed_dim,
                    ") must be divisible by elems_per_thread (", elems_per_thread, ")");
    }

    dim3 grid((total_warps + rdna2::WARPS_PER_BLOCK - 1) / rdna2::WARPS_PER_BLOCK);
    dim3 block(rdna2::BLOCK_SIZE);
    auto stream = at::hip::getCurrentHIPStream();

    #define LAUNCH_FUSED_QKNORM_ROPE(HEAD_DIM_VAL, NEOX_VAL) \
        AT_DISPATCH_FLOATING_TYPES_AND2( \
            at::ScalarType::Half, at::ScalarType::BFloat16, \
            qkv.scalar_type(), "rdna2_fused_qknorm_rope", [&] { \
                rdna2::fused_qknorm_rope_kernel<scalar_t, HEAD_DIM_VAL, NEOX_VAL> \
                    <<<grid, block, 0, stream>>>( \
                    qkv.data_ptr<scalar_t>(), \
                    q_weight.data_ptr<scalar_t>(), \
                    k_weight.data_ptr<scalar_t>(), \
                    cos_sin_cache.data_ptr<scalar_t>(), \
                    positions.data_ptr<int64_t>(), \
                    num_heads_q, num_heads_k, num_heads_v, \
                    rotary_dim, eps, num_tokens); \
            })

    if (is_neox) {
        switch (head_dim) {
            case 64:  LAUNCH_FUSED_QKNORM_ROPE(64, true); break;
            case 128: LAUNCH_FUSED_QKNORM_ROPE(128, true); break;
            case 256: LAUNCH_FUSED_QKNORM_ROPE(256, true); break;
            default:
                TORCH_CHECK(false, "Unsupported head_dim=", head_dim,
                            " for fused QKNorm+RoPE. Supported: 64, 128, 256");
        }
    } else {
        switch (head_dim) {
            case 64:  LAUNCH_FUSED_QKNORM_ROPE(64, false); break;
            case 128: LAUNCH_FUSED_QKNORM_ROPE(128, false); break;
            case 256: LAUNCH_FUSED_QKNORM_ROPE(256, false); break;
            default:
                TORCH_CHECK(false, "Unsupported head_dim=", head_dim,
                            " for fused QKNorm+RoPE. Supported: 64, 128, 256");
        }
    }

    #undef LAUNCH_FUSED_QKNORM_ROPE
}
"""

_compiled_module = None


def _get_module():
    """Lazily compile and cache the fused QKNorm+RoPE HIP kernel."""
    global _compiled_module
    if _compiled_module is not None:
        return _compiled_module

    try:
        from torch.utils.cpp_extension import load_inline

        _compiled_module = load_inline(
            name="rdna2_fused_qknorm_rope",
            cpp_sources=RDNA2_FUSED_QKNORM_ROPE_DECL,
            cuda_sources=RDNA2_FUSED_QKNORM_ROPE_CU,
            functions=["rdna2_fused_qknorm_rope"],
            extra_cuda_cflags=[
                "--offload-arch=gfx1030",
                "-O3",
                "-mno-wavefrontsize64",
            ],
            verbose=False,
        )
        logger.info("RDNA2 fused QKNorm+RoPE: compiled via torch cpp_extension")
    except Exception as e:
        logger.warning(f"Fused QKNorm+RoPE compilation failed: {e}")
        _compiled_module = None

    return _compiled_module


def can_use_fused_qknorm_rope(head_dim: int, dtype: torch.dtype) -> bool:
    """Check if the RDNA2 fused QKNorm+RoPE kernel is available."""
    if head_dim not in (64, 128, 256):
        return False
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    try:
        return _get_module() is not None
    except Exception:
        return False


def fused_qknorm_rope(
    qkv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos_sin_cache: Tensor,
    position_ids: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    rotary_dim: int,
    eps: float = 1e-6,
    is_neox: bool = True,
) -> None:
    """Fused per-head QKNorm + RoPE, in-place on QKV."""
    mod = _get_module()
    if mod is not None:
        mod.rdna2_fused_qknorm_rope(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            position_ids,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            rotary_dim,
            eps,
            is_neox,
        )
        return

    _fallback_qknorm_rope(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        position_ids,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        is_neox,
    )


def _fallback_qknorm_rope(
    qkv,
    q_weight,
    k_weight,
    cos_sin_cache,
    position_ids,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    rotary_dim,
    eps,
    is_neox,
):
    """Pure PyTorch fallback — correct but slower."""
    num_tokens = qkv.shape[0]

    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim

    q = qkv[:, :q_size].view(num_tokens, num_heads_q, head_dim)
    k = qkv[:, q_size : q_size + k_size].view(num_tokens, num_heads_k, head_dim)

    def _rms_norm(x, weight):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return (x_normed * weight).to(x.dtype)

    q_normed = _rms_norm(q, q_weight)
    k_normed = _rms_norm(k, k_weight)

    embed_dim = rotary_dim // 2
    cos_sin = cos_sin_cache[position_ids]
    cos_vals = cos_sin[:, :embed_dim].unsqueeze(1)
    sin_vals = cos_sin[:, embed_dim:].unsqueeze(1)

    def _apply_rope_neox(x):
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x0 = x_rot[..., :embed_dim]
        x1 = x_rot[..., embed_dim:]
        rotated = torch.cat(
            [x0 * cos_vals - x1 * sin_vals, x1 * cos_vals + x0 * sin_vals], dim=-1
        )
        return torch.cat([rotated, x_pass], dim=-1) if x_pass.numel() > 0 else rotated

    def _apply_rope_interleave(x):
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x0 = x_rot[..., 0::2]
        x1 = x_rot[..., 1::2]
        rotated = torch.stack(
            [x0 * cos_vals - x1 * sin_vals, x0 * sin_vals + x1 * cos_vals], dim=-1
        ).flatten(-2)
        return torch.cat([rotated, x_pass], dim=-1) if x_pass.numel() > 0 else rotated

    apply_rope = _apply_rope_neox if is_neox else _apply_rope_interleave
    q_final = apply_rope(q_normed).view(num_tokens, -1)
    k_final = apply_rope(k_normed).view(num_tokens, -1)

    qkv[:, :q_size] = q_final
    qkv[:, q_size : q_size + k_size] = k_final
