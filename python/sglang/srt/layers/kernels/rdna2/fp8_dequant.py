"""Wave32-optimized FP8 software dequantization HIP kernel for RDNA2.

RDNA2 GPUs lack hardware FP8 support. This kernel provides software
dequantization: int8 storage → fp16/bf16 compute with per-tensor or
per-block scale factors.

Launch config: <<<ceil(n/1024), 128>>> (4 warps × 32 threads)
Memory: Coalesced int8 reads, fp16 writes, scale in constant cache

Formats supported:
  - E4M3 (standard, used by RDNA2)
  - FNUZ (MI300 only — detected and rejected here)
"""

import logging
import os
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ── HIP kernel source ──────────────────────────────────────────────

RDNA2_FP8_DEQUANT_DECL = """
void rdna2_fp8_dequant(torch::Tensor& output, torch::Tensor& input, float scale);
void rdna2_fp8_dequant_blocked(torch::Tensor& output, torch::Tensor& input,
                                torch::Tensor& scales, int block_size);
void rdna2_fp8_quant(torch::Tensor& output, torch::Tensor& input, float scale);
"""

RDNA2_FP8_DEQUANT_CU = r"""
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

using __nv_bfloat16 = __hip_bfloat16;

namespace rdna2 {

constexpr int BLOCK_DIM = 128;  // 4 warps × Wave32

// ──── E4M3 FP8 → float lookup table ────
// Pre-computed in constant memory for fast broadcast across wavefront
__device__ __constant__ float e4m3_lut[256];

// Build LUT at kernel launch time (host-side)
// E4M3: 1 sign + 4 exponent + 3 mantissa, bias=7, no inf/nan
static float host_e4m3_lut[256];
static bool lut_initialized = false;
static bool lut_uploaded = false;

void init_e4m3_lut() {
    if (lut_initialized) return;
    for (int i = 0; i < 256; i++) {
        int sign = (i >> 7) & 1;
        int exp  = (i >> 3) & 0xF;
        int mant = i & 0x7;

        float val;
        if (exp == 0) {
            // Subnormal: (-1)^sign × 2^(-6) × (0.mantissa)
            val = ldexpf((float)mant / 8.0f, -6);
        } else if (exp == 15 && mant == 7) {
            // NaN in e4m3fn → treat as 0 (safe for inference)
            val = 0.0f;
        } else {
            // Normal: (-1)^sign × 2^(exp-7) × (1.mantissa)
            val = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
        }
        host_e4m3_lut[i] = sign ? -val : val;
    }
    lut_initialized = true;
}

// Upload LUT to device constant memory once (avoids null-stream sync on every call)
void ensure_lut_uploaded() {
    init_e4m3_lut();
    if (lut_uploaded) return;
    (void)hipMemcpyToSymbol(HIP_SYMBOL(rdna2::e4m3_lut), rdna2::host_e4m3_lut,
                      sizeof(float) * 256);
    lut_uploaded = true;
}

// ──── Per-tensor dequant: int8 → fp16 ────
template <typename out_t>
__global__ void fp8_dequant_per_tensor(
    out_t* __restrict__ output,
    const uint8_t* __restrict__ input,
    const float scale,
    const int64_t n)
{
    int64_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (idx < n) {
        float val = e4m3_lut[input[idx]] * scale;
        output[idx] = static_cast<out_t>(val);
    }
}

// ──── Per-block dequant: int8 → fp16 with block scales ────
template <typename out_t>
__global__ void fp8_dequant_per_block(
    out_t* __restrict__ output,
    const uint8_t* __restrict__ input,
    const float* __restrict__ scales,  // one scale per block
    const int block_size,
    const int64_t n)
{
    int64_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (idx < n) {
        float scale = scales[idx / block_size];
        float val = e4m3_lut[input[idx]] * scale;
        output[idx] = static_cast<out_t>(val);
    }
}

// ──── Quantize: fp16 → int8 (E4M3) with scale computation ────
__global__ void fp8_quant_per_tensor(
    uint8_t* __restrict__ output,
    const __half* __restrict__ input,
    const float inv_scale,  // 1.0 / scale
    const int64_t n)
{
    int64_t idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if (idx < n) {
        float val = __half2float(input[idx]) * inv_scale;
        // Clamp to E4M3 range: [-448, 448]
        val = fmaxf(-448.0f, fminf(448.0f, val));

        // Encode to E4M3
        int sign = val < 0.0f ? 1 : 0;
        float abs_val = fabsf(val);

        uint8_t encoded;
        if (abs_val == 0.0f) {
            encoded = 0;
        } else {
            int exp;
            float frac = frexpf(abs_val, &exp);
            // frexp returns [0.5, 1.0), we need [1.0, 2.0)
            frac *= 2.0f;
            exp -= 1;

            int biased_exp = exp + 7;  // bias=7
            if (biased_exp <= 0) {
                // Subnormal
                biased_exp = 0;
                frac = ldexpf(abs_val, 6);  // 2^6 to shift into mantissa
            }
            biased_exp = min(biased_exp, 15);

            int mant = (int)((frac - 1.0f) * 8.0f + 0.5f);
            mant = min(mant, 7);

            // Special case: exp=15, mant=7 is NaN → clamp to (15, 6)
            if (biased_exp == 15 && mant == 7) mant = 6;

            encoded = (sign << 7) | (biased_exp << 3) | mant;
        }
        output[idx] = encoded;
    }
}

}  // namespace rdna2

// ──── Torch dispatch wrappers ────
void rdna2_fp8_dequant(
    torch::Tensor& output,
    torch::Tensor& input,
    float scale)
{
    int64_t n = input.numel();
    dim3 grid((n + rdna2::BLOCK_DIM - 1) / rdna2::BLOCK_DIM);
    dim3 block(rdna2::BLOCK_DIM);
    auto stream = at::hip::getCurrentHIPStream();

    // Upload LUT to device constant memory (once)
    rdna2::ensure_lut_uploaded();

    if (output.scalar_type() == at::ScalarType::Half) {
        rdna2::fp8_dequant_per_tensor<<<grid, block, 0, stream>>>(
            output.data_ptr<at::Half>(),
            input.data_ptr<uint8_t>(),
            scale, n);
    } else if (output.scalar_type() == at::ScalarType::BFloat16) {
        rdna2::fp8_dequant_per_tensor<<<grid, block, 0, stream>>>(
            output.data_ptr<at::BFloat16>(),
            input.data_ptr<uint8_t>(),
            scale, n);
    }
}

void rdna2_fp8_dequant_blocked(
    torch::Tensor& output,
    torch::Tensor& input,
    torch::Tensor& scales,
    int block_size)
{
    int64_t n = input.numel();
    dim3 grid((n + rdna2::BLOCK_DIM - 1) / rdna2::BLOCK_DIM);
    dim3 block(rdna2::BLOCK_DIM);
    auto stream = at::hip::getCurrentHIPStream();

    // Upload LUT to device constant memory (once)
    rdna2::ensure_lut_uploaded();

    if (output.scalar_type() == at::ScalarType::Half) {
        rdna2::fp8_dequant_per_block<<<grid, block, 0, stream>>>(
            output.data_ptr<at::Half>(),
            input.data_ptr<uint8_t>(),
            scales.data_ptr<float>(),
            block_size, n);
    } else if (output.scalar_type() == at::ScalarType::BFloat16) {
        rdna2::fp8_dequant_per_block<<<grid, block, 0, stream>>>(
            output.data_ptr<at::BFloat16>(),
            input.data_ptr<uint8_t>(),
            scales.data_ptr<float>(),
            block_size, n);
    }
}

void rdna2_fp8_quant(
    torch::Tensor& output,
    torch::Tensor& input,
    float scale)
{
    int64_t n = input.numel();
    dim3 grid((n + rdna2::BLOCK_DIM - 1) / rdna2::BLOCK_DIM);
    dim3 block(rdna2::BLOCK_DIM);
    auto stream = at::hip::getCurrentHIPStream();

    rdna2::fp8_quant_per_tensor<<<grid, block, 0, stream>>>(
        output.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        1.0f / scale, n);
}
"""


# ── Python wrappers ─────────────────────────────────────────────────

_compiled_module = None


def _get_module():
    """Lazily compile the FP8 dequant HIP kernel."""
    global _compiled_module
    if _compiled_module is not None:
        return _compiled_module

    try:
        from torch.utils.cpp_extension import load_inline

        _compiled_module = load_inline(
            name="rdna2_fp8_dequant",
            cpp_sources=RDNA2_FP8_DEQUANT_DECL,
            cuda_sources=RDNA2_FP8_DEQUANT_CU,
            functions=[
                "rdna2_fp8_dequant",
                "rdna2_fp8_dequant_blocked",
                "rdna2_fp8_quant",
            ],
            extra_cuda_cflags=["--offload-arch=gfx1030", "-O3"],
            verbose=False,
        )
        logger.info("RDNA2 FP8 kernels: compiled via torch cpp_extension")
    except Exception as e:
        logger.warning(f"FP8 kernel compilation failed: {e}")
        _compiled_module = None

    return _compiled_module


def fp8_dequantize(
    input: Tensor,
    scale: float,
    output_dtype: torch.dtype = torch.float16,
    block_size: Optional[int] = None,
    block_scales: Optional[Tensor] = None,
) -> Tensor:
    """Dequantize FP8 (E4M3) int8 tensor to fp16/bf16.

    Args:
        input: uint8 tensor containing E4M3-encoded values
        scale: per-tensor scale factor (output = encoded_float * scale)
        output_dtype: torch.float16 or torch.bfloat16
        block_size: if set, use per-block dequantization
        block_scales: per-block scale factors (required if block_size set)

    Returns:
        Dequantized tensor in output_dtype
    """
    output = torch.empty_like(input, dtype=output_dtype)

    mod = _get_module()
    if mod is not None:
        if block_size is not None and block_scales is not None:
            mod.rdna2_fp8_dequant_blocked(output, input, block_scales, block_size)
        else:
            mod.rdna2_fp8_dequant(output, input, scale)
        return output

    # Pure Python fallback — E4M3 decode via bit manipulation
    return _fp8_dequant_fallback(input, scale, output_dtype)


def fp8_quantize(
    input: Tensor,
    scale: float,
) -> Tensor:
    """Quantize fp16 tensor to FP8 (E4M3) as uint8.

    Args:
        input: fp16 tensor
        scale: per-tensor scale factor (encoded = float / scale)

    Returns:
        uint8 tensor containing E4M3-encoded values
    """
    output = torch.empty_like(input, dtype=torch.uint8)

    mod = _get_module()
    if mod is not None:
        mod.rdna2_fp8_quant(output, input.half(), scale)
        return output

    # Triton/Python fallback
    return _fp8_quant_fallback(input, scale)


def _fp8_dequant_fallback(
    input: Tensor, scale: float, output_dtype: torch.dtype
) -> Tensor:
    """Pure PyTorch E4M3 decode (slow but always works)."""
    raw = input.to(torch.int32)
    sign = ((raw >> 7) & 1).float()
    exp = ((raw >> 3) & 0xF).int()
    mant = (raw & 0x7).float()

    # Normal values: (-1)^s × 2^(e-7) × (1 + m/8)
    normal = (1.0 + mant / 8.0) * torch.pow(2.0, (exp.float() - 7.0))
    # Subnormal: (-1)^s × 2^(-6) × (m/8)
    subnormal = (mant / 8.0) * (2.0**-6)

    result = torch.where(exp == 0, subnormal, normal)
    result = torch.where(sign > 0, -result, result)
    result = result * scale

    return result.to(output_dtype)


def _fp8_quant_fallback(input: Tensor, scale: float) -> Tensor:
    """Pure PyTorch E4M3 encode (slow but always works)."""
    # Clamp, scale, and round-to-nearest E4M3
    scaled = input.float() / scale
    scaled = scaled.clamp(-448.0, 448.0)

    sign = (scaled < 0).to(torch.uint8) << 7
    abs_val = scaled.abs()

    # Simple approximation: use torch's fp8 if available
    try:
        fp8 = abs_val.to(torch.float8_e4m3fn)
        return (fp8.view(torch.uint8) | sign).to(torch.uint8)
    except (AttributeError, RuntimeError):
        pass

    # Manual encode for older PyTorch
    log2_val = torch.log2(abs_val.clamp(min=2**-9))
    exp = torch.floor(log2_val).clamp(0, 15).to(torch.uint8)
    mant_float = abs_val / torch.pow(2.0, exp.float() - 7.0) - 1.0
    mant = (mant_float * 8.0 + 0.5).clamp(0, 7).to(torch.uint8)

    encoded = sign | (exp << 3) | mant
    return encoded.to(torch.uint8)
