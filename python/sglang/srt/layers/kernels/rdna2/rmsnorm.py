"""Wave32-optimized RMSNorm HIP kernel for RDNA2 (gfx1030).

Launch config: <<<num_tokens, 128>>> (4 warps × 32 threads)
Memory access: vec8 coalesced loads (8 elements per thread per iteration)
Reduction: hipcub BlockReduce → shared memory broadcast

Adapted from AITER's rmsnorm_kernels.cu with Wave32-specific tuning:
  - 128 threads/block instead of 1024 (RDNA2 occupancy sweet spot)
  - hipcub BlockReduce<float, 128> for exact thread count
  - vec8 loads aligned to 16 bytes for L2 cache line utilization
"""

import logging
import os
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ── HIP kernel source ──────────────────────────────────────────────
# This is compiled at first call via AITER JIT or torch cpp_extension.

RDNA2_RMSNORM_DECL = """
void rdna2_rms_norm(torch::Tensor& out, torch::Tensor& input,
                    torch::Tensor& weight, float epsilon);
void rdna2_fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                               torch::Tensor& weight, float epsilon);
"""

RDNA2_RMSNORM_CU = r"""
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

using __nv_bfloat16 = __hip_bfloat16;

namespace rdna2 {

// vec8: 8-wide vector for coalesced 16-byte loads on RDNA2
// Uses explicit constructor instead of brace-init for HIP/clang compat
template <typename scalar_t>
struct __align__(16) vec8_t {
    scalar_t x, y, z, w, u, v, s, t;

    __device__ vec8_t() : x(0), y(0), z(0), w(0), u(0), v(0), s(0), t(0) {}
    __device__ vec8_t(scalar_t a, scalar_t b, scalar_t c, scalar_t d,
                      scalar_t e, scalar_t f, scalar_t g, scalar_t h)
        : x(a), y(b), z(c), w(d), u(e), v(f), s(g), t(h) {}

    __device__ vec8_t operator*(const vec8_t& o) const {
        return vec8_t(x*o.x, y*o.y, z*o.z, w*o.w, u*o.u, v*o.v, s*o.s, t*o.t);
    }
    __device__ vec8_t operator*(const float& sc) const {
        return vec8_t(scalar_t(float(x)*sc), scalar_t(float(y)*sc),
                      scalar_t(float(z)*sc), scalar_t(float(w)*sc),
                      scalar_t(float(u)*sc), scalar_t(float(v)*sc),
                      scalar_t(float(s)*sc), scalar_t(float(t)*sc));
    }
    __device__ vec8_t operator+(const vec8_t& o) const {
        return vec8_t(x+o.x, y+o.y, z+o.z, w+o.w, u+o.u, v+o.v, s+o.s, t+o.t);
    }
    __device__ float sum() const {
        return float(x)+float(y)+float(z)+float(w)+
               float(u)+float(v)+float(s)+float(t);
    }
};

// ──── Wave32-tuned RMSNorm ────
// 128 threads = 4 warps × 32 (Wave32)
// BlockReduce with exactly 128 threads for hipcub efficiency
constexpr int BLOCK_DIM = 128;

template <typename scalar_t>
__global__ void rms_norm_wave32(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{
    __shared__ float s_variance;

    const int vec_hidden = hidden_size >> 3;
    const auto* vec_in = reinterpret_cast<const vec8_t<scalar_t>*>(input);
    const auto* vec_w  = reinterpret_cast<const vec8_t<scalar_t>*>(weight);
    auto* vec_out      = reinterpret_cast<vec8_t<scalar_t>*>(out);

    // Phase 1: Compute sum of squares via vec8 loads
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden; i += BLOCK_DIM) {
        vec8_t<scalar_t> v = vec_in[blockIdx.x * vec_hidden + i];
        local_ss += (v * v).sum();
    }

    // Phase 2: Wave32-efficient block reduce
    // Use warp shuffle first (32-thread waves), then shared mem across warps
    for (int offset = 16; offset > 0; offset >>= 1)
        local_ss += __shfl_xor(local_ss, offset);

    // Cross-warp reduce via shared memory (4 warps)
    __shared__ float warp_sums[4];
    int warp_id = threadIdx.x >> 5;  // / 32
    int lane_id = threadIdx.x & 31;  // % 32

    if (lane_id == 0) warp_sums[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0 && lane_id < 4) {
        float val = warp_sums[lane_id];
        for (int offset = 2; offset > 0; offset >>= 1)
            val += __shfl_xor(val, offset);
        if (lane_id == 0)
            s_variance = rsqrtf(val / hidden_size + epsilon);
    }
    __syncthreads();

    // Phase 3: Apply normalization with weight
    for (int i = threadIdx.x; i < vec_hidden; i += BLOCK_DIM) {
        vec8_t<scalar_t> v = vec_in[blockIdx.x * vec_hidden + i];
        vec8_t<scalar_t> w = vec_w[i];
        vec_out[blockIdx.x * vec_hidden + i] = (v * s_variance) * w;
    }
}

// ──── Fused Add + RMSNorm (residual connection) ────
template <typename scalar_t>
__global__ void fused_add_rms_norm_wave32(
    scalar_t* __restrict__ input,       // in/out
    scalar_t* __restrict__ residual,    // in/out
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const int num_tokens,
    const int hidden_size)
{
    __shared__ float s_variance;

    const int vec_hidden = hidden_size >> 3;
    auto* vec_inp = reinterpret_cast<vec8_t<scalar_t>*>(input);
    auto* vec_res = reinterpret_cast<vec8_t<scalar_t>*>(residual);
    const auto* vec_w = reinterpret_cast<const vec8_t<scalar_t>*>(weight);

    // Phase 1: Fuse add + compute variance
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden; i += BLOCK_DIM) {
        int idx = blockIdx.x * vec_hidden + i;
        vec8_t<scalar_t> inp_v = vec_inp[idx];
        vec8_t<scalar_t> res_v = vec_res[idx];
        vec8_t<scalar_t> sum_v = inp_v + res_v;
        vec_res[idx] = sum_v;  // write residual
        local_ss += (sum_v * sum_v).sum();
    }

    // Phase 2: Wave32 reduce (same as above)
    for (int offset = 16; offset > 0; offset >>= 1)
        local_ss += __shfl_xor(local_ss, offset);

    __shared__ float warp_sums[4];
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    if (lane_id == 0) warp_sums[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0 && lane_id < 4) {
        float val = warp_sums[lane_id];
        for (int offset = 2; offset > 0; offset >>= 1)
            val += __shfl_xor(val, offset);
        if (lane_id == 0)
            s_variance = rsqrtf(val / hidden_size + epsilon);
    }
    __syncthreads();

    // Phase 3: Normalize from residual, write to input
    for (int i = threadIdx.x; i < vec_hidden; i += BLOCK_DIM) {
        int idx = blockIdx.x * vec_hidden + i;
        vec8_t<scalar_t> v = vec_res[idx];
        vec8_t<scalar_t> w = vec_w[i];
        vec_inp[idx] = (v * s_variance) * w;
    }
}

}  // namespace rdna2

// ──── Torch dispatch wrappers ────
void rdna2_rms_norm(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon)
{
    int num_tokens = input.size(0);
    int hidden_size = input.size(-1);

    dim3 grid(num_tokens);
    dim3 block(rdna2::BLOCK_DIM);
    auto stream = at::hip::getCurrentHIPStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rdna2_rms_norm", [&] {
            rdna2::rms_norm_wave32<<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                epsilon, num_tokens, hidden_size);
        });
}

void rdna2_fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    float epsilon)
{
    int num_tokens = input.size(0);
    int hidden_size = input.size(-1);

    dim3 grid(num_tokens);
    dim3 block(rdna2::BLOCK_DIM);
    auto stream = at::hip::getCurrentHIPStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rdna2_fused_add_rms_norm", [&] {
            rdna2::fused_add_rms_norm_wave32<<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                epsilon, num_tokens, hidden_size);
        });
}
"""


# ── Python wrappers ─────────────────────────────────────────────────

# Compiled module cache
_compiled_module = None


def _get_module():
    """Lazily compile and cache the RDNA2 RMSNorm HIP kernel."""
    global _compiled_module
    if _compiled_module is not None:
        return _compiled_module

    from . import _backend, _aiter_available

    if _aiter_available:
        # AITER JIT path — register as a custom module
        try:
            from aiter.jit.core import _jit_compile
            import tempfile

            src_path = os.path.join(
                os.path.dirname(__file__), "_hip_src", "rdna2_rmsnorm.cu"
            )
            # If we have the .cu on disk, use AITER. Otherwise use inline.
            if os.path.exists(src_path):
                _compiled_module = _jit_compile(
                    "rdna2_rmsnorm",
                    [src_path],
                    extra_cflags=[],
                    extra_hip_flags=["--offload-arch=gfx1030"],
                )
                logger.info("RDNA2 RMSNorm: compiled via AITER JIT")
                return _compiled_module
        except Exception as e:
            logger.warning(f"AITER JIT failed for RMSNorm: {e}, falling back")

    # torch cpp_extension inline path
    try:
        from torch.utils.cpp_extension import load_inline

        _compiled_module = load_inline(
            name="rdna2_rmsnorm",
            cpp_sources=RDNA2_RMSNORM_DECL,
            cuda_sources=RDNA2_RMSNORM_CU,
            functions=["rdna2_rms_norm", "rdna2_fused_add_rms_norm"],
            extra_cuda_cflags=["--offload-arch=gfx1030", "-O3"],
            verbose=False,
        )
        logger.info("RDNA2 RMSNorm: compiled via torch cpp_extension")
    except Exception as e:
        logger.warning(f"torch cpp_extension failed for RMSNorm: {e}")
        _compiled_module = None

    return _compiled_module


def rms_norm(
    out: Tensor, input: Tensor, weight: Tensor, epsilon: float = 1e-6
) -> Tensor:
    """Wave32-optimized RMSNorm.

    Falls back to Triton or PyTorch if HIP compilation fails.
    """
    mod = _get_module()
    if mod is not None:
        mod.rdna2_rms_norm(out, input, weight, epsilon)
        return out

    # Triton fallback — rmsnorm_autotune is a @triton.autotune kernel,
    # not a Python function. Use the layernorm module's fallback instead.
    try:
        from sglang.srt.layers.layernorm import RMSNorm

        variance = input.float().pow(2).mean(-1, keepdim=True)
        normed = input * torch.rsqrt(variance + epsilon)
        out.copy_(normed * weight)
        return out
    except ImportError:
        pass

    # PyTorch native fallback
    variance = input.float().pow(2).mean(-1, keepdim=True)
    normed = input * torch.rsqrt(variance + epsilon)
    out.copy_(normed * weight)
    return out


def fused_add_rms_norm(
    input: Tensor, residual: Tensor, weight: Tensor, epsilon: float = 1e-6
) -> tuple:
    """Wave32-optimized fused residual add + RMSNorm.

    Returns (normalized_output, updated_residual).
    Falls back to sequential add + norm if HIP compilation fails.
    """
    mod = _get_module()
    if mod is not None:
        mod.rdna2_fused_add_rms_norm(input, residual, weight, epsilon)
        return input, residual

    # Fallback: sequential add + norm
    residual.add_(input)
    variance = residual.float().pow(2).mean(-1, keepdim=True)
    normed = residual * torch.rsqrt(variance + epsilon)
    input.copy_(normed * weight)
    return input, residual
