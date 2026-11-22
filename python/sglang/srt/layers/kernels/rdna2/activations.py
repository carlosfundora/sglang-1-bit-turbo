"""Wave32-optimized activation function HIP kernels for RDNA2.

Provides fused SiLU (Swish) and GELU activation kernels with
optional gating (SiLU-gate used by Llama/Qwen FFN layers).

Launch config: <<<ceil(n/1024), 128>>> (4 warps × Wave32)
Fused gate: reads gate+up in one pass, avoids extra kernel launch.
"""

import logging
import os

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

RDNA2_ACTIVATIONS_DECL = """
void rdna2_silu(torch::Tensor& out, torch::Tensor& input);
void rdna2_silu_and_mul(torch::Tensor& out, torch::Tensor& input, int d);
void rdna2_gelu(torch::Tensor& out, torch::Tensor& input);
void rdna2_gelu_and_mul(torch::Tensor& out, torch::Tensor& input, int d);
"""

RDNA2_ACTIVATIONS_CU = r"""
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

using __nv_bfloat16 = __hip_bfloat16;

namespace rdna2 {

constexpr int BLOCK_DIM = 128;

// ──── SiLU (Swish): x * sigmoid(x) ────
template <typename scalar_t>
__global__ void silu_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const int64_t n)
{
    int64_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(input[idx]);
        float result = x / (1.0f + expf(-x));
        out[idx] = static_cast<scalar_t>(result);
    }
}

// ──── Fused SiLU-Gate: silu(gate) * up ────
// Gate and up are concatenated: input[..., :d] = gate, input[..., d:] = up
template <typename scalar_t>
__global__ void silu_and_mul_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const int d,         // half of last dimension
    const int64_t n)     // total output elements
{
    int64_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (idx < n) {
        int row = idx / d;
        int col = idx % d;
        float gate = static_cast<float>(input[row * 2 * d + col]);
        float up   = static_cast<float>(input[row * 2 * d + d + col]);
        float result = (gate / (1.0f + expf(-gate))) * up;
        out[idx] = static_cast<scalar_t>(result);
    }
}

// ──── GELU (approximate): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) ────
template <typename scalar_t>
__global__ void gelu_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const int64_t n)
{
    int64_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(input[idx]);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[idx] = static_cast<scalar_t>(x * cdf);
    }
}

// ──── Fused GELU-Gate: gelu(gate) * up ────
template <typename scalar_t>
__global__ void gelu_and_mul_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const int d,
    const int64_t n)
{
    int64_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (idx < n) {
        int row = idx / d;
        int col = idx % d;
        float gate = static_cast<float>(input[row * 2 * d + col]);
        float up   = static_cast<float>(input[row * 2 * d + d + col]);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (gate + 0.044715f * gate * gate * gate)));
        out[idx] = static_cast<scalar_t>(gate * cdf * up);
    }
}

}  // namespace rdna2

// ──── Torch wrappers ────
void rdna2_silu(torch::Tensor& out, torch::Tensor& input)
{
    int64_t n = input.numel();
    dim3 grid((n + rdna2::BLOCK_DIM - 1) / rdna2::BLOCK_DIM);
    dim3 block(rdna2::BLOCK_DIM);
    auto stream = at::hip::getCurrentHIPStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rdna2_silu", [&] {
            rdna2::silu_kernel<<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), n);
        });
}

void rdna2_silu_and_mul(torch::Tensor& out, torch::Tensor& input, int d)
{
    int64_t n = out.numel();
    dim3 grid((n + rdna2::BLOCK_DIM - 1) / rdna2::BLOCK_DIM);
    dim3 block(rdna2::BLOCK_DIM);
    auto stream = at::hip::getCurrentHIPStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rdna2_silu_and_mul", [&] {
            rdna2::silu_and_mul_kernel<<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d, n);
        });
}

void rdna2_gelu(torch::Tensor& out, torch::Tensor& input)
{
    int64_t n = input.numel();
    dim3 grid((n + rdna2::BLOCK_DIM - 1) / rdna2::BLOCK_DIM);
    dim3 block(rdna2::BLOCK_DIM);
    auto stream = at::hip::getCurrentHIPStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rdna2_gelu", [&] {
            rdna2::gelu_kernel<<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), n);
        });
}

void rdna2_gelu_and_mul(torch::Tensor& out, torch::Tensor& input, int d)
{
    int64_t n = out.numel();
    dim3 grid((n + rdna2::BLOCK_DIM - 1) / rdna2::BLOCK_DIM);
    dim3 block(rdna2::BLOCK_DIM);
    auto stream = at::hip::getCurrentHIPStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rdna2_gelu_and_mul", [&] {
            rdna2::gelu_and_mul_kernel<<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d, n);
        });
}
"""

_compiled_module = None


def _get_module():
    """Lazily compile the activation HIP kernels."""
    global _compiled_module
    if _compiled_module is not None:
        return _compiled_module

    try:
        from torch.utils.cpp_extension import load_inline

        _compiled_module = load_inline(
            name="rdna2_activations",
            cpp_sources=RDNA2_ACTIVATIONS_DECL,
            cuda_sources=RDNA2_ACTIVATIONS_CU,
            functions=[
                "rdna2_silu",
                "rdna2_silu_and_mul",
                "rdna2_gelu",
                "rdna2_gelu_and_mul",
            ],
            extra_cuda_cflags=["--offload-arch=gfx1030", "-O3"],
            verbose=False,
        )
        logger.info("RDNA2 activation kernels: compiled")
    except Exception as e:
        logger.warning(f"Activation kernel compilation failed: {e}")
        _compiled_module = None

    return _compiled_module


def silu(input: Tensor) -> Tensor:
    """Wave32-optimized SiLU activation."""
    out = torch.empty_like(input)
    mod = _get_module()
    if mod is not None:
        mod.rdna2_silu(out, input)
        return out
    return torch.nn.functional.silu(input)


def silu_and_mul(input: Tensor) -> Tensor:
    """Fused SiLU-gate: silu(gate) * up.

    Input shape: [..., 2*d] where first half is gate, second is up.
    Output shape: [..., d]
    """
    d = input.shape[-1] // 2
    out = torch.empty(
        *input.shape[:-1], d, dtype=input.dtype, device=input.device
    )
    mod = _get_module()
    if mod is not None:
        mod.rdna2_silu_and_mul(out, input, d)
        return out
    # PyTorch fallback
    gate = input[..., :d]
    up = input[..., d:]
    return torch.nn.functional.silu(gate) * up


def gelu(input: Tensor) -> Tensor:
    """Wave32-optimized approximate GELU."""
    out = torch.empty_like(input)
    mod = _get_module()
    if mod is not None:
        mod.rdna2_gelu(out, input)
        return out
    return torch.nn.functional.gelu(input, approximate="tanh")


def gelu_and_mul(input: Tensor) -> Tensor:
    """Fused GELU-gate: gelu(gate) * up."""
    d = input.shape[-1] // 2
    out = torch.empty(
        *input.shape[:-1], d, dtype=input.dtype, device=input.device
    )
    mod = _get_module()
    if mod is not None:
        mod.rdna2_gelu_and_mul(out, input, d)
        return out
    gate = input[..., :d]
    up = input[..., d:]
    return torch.nn.functional.gelu(gate, approximate="tanh") * up
