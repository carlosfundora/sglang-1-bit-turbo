# RDNA2 (gfx1030) Support — SGLang 1-Bit Turbo

This document describes the RDNA2 GPU support infrastructure in the
`sglang-1-bit-turbo` fork.

## Supported Hardware

All RDNA2 GPUs (gfx1030–gfx1036) are supported via the gfx1030 ISA target:

| GPU | Architecture | Status |
|-----|-------------|--------|
| RX 6900 XT | gfx1030 | ✅ Native target |
| RX 6800 XT | gfx1030 | ✅ Native target |
| RX 6800 | gfx1030 | ✅ Native target |
| RX 6700 XT | gfx1031 | ✅ via HSA_OVERRIDE_GFX_VERSION |
| RX 6700 | gfx1031 | ✅ via HSA_OVERRIDE_GFX_VERSION |
| RX 6650 XT | gfx1032 | ✅ via HSA_OVERRIDE_GFX_VERSION |
| RX 6600 XT | gfx1032 | ✅ via HSA_OVERRIDE_GFX_VERSION |
| RX 6600 | gfx1032 | ✅ via HSA_OVERRIDE_GFX_VERSION |
| RX 6500 XT | gfx1034 | ⚠️ 4 GB VRAM limit |

## Quick Start

```bash
# 1. Set up the environment
source scripts/setup_rocm_gfx1030.sh

# 2. Verify compatibility
python scripts/check_rocm_gfx1030.py

# 3. Launch SGLang (auto-detects RDNA2 and applies optimal settings)
python -m sglang.launch_server \
  --model-path your-model \
  --mem-fraction-static 0.35 \
  --attention-backend torch_native \
  --disable-cuda-graph
```

## Architecture Overview

### Hardware Backend Module

`python/sglang/srt/hardware_backend/rocm/`

| File | Purpose |
|------|---------|
| `arch_detection.py` | GPU architecture detection (env → PyTorch → rocminfo) |
| `gfx1031_defaults.py` | Server defaults, env setup, Wave32 tuning configs |
| `te_integration.py` | Optional TransformerEngine hooks (RMSNorm, FP8) |

### Kernel Tuning

RDNA2 uses **Wave32** (32 threads/wavefront) instead of CDNA's Wave64.
This requires different Triton kernel parameters:

| Parameter | CDNA (MI250/MI300) | RDNA2 (gfx1030) |
|-----------|-------------------|------------------|
| Wavefront size | 64 | 32 |
| Typical num_warps | 4–8 | 2–4 |
| BLOCK_M (attention) | 64 | 32 |
| waves_per_eu | 1–2 | 2–4 |
| matrix_instr_nonkdim | 16 | N/A (no matrix cores) |

### Auto-Detection

When SGLang starts on an RDNA2 GPU, `server_args.py` automatically:
1. Detects the GPU architecture via `arch_detection.get_rocm_arch()`
2. Sets `HSA_OVERRIDE_GFX_VERSION=10.3.0` for ISA compatibility
3. Configures `attention_backend=triton` (or `torch_native` if preferred)
4. Applies Wave32-optimized CUDA graph batch sizes
5. Sets `kv_splits=8` for Wave32-optimal KV splitting

### FP8 on RDNA2

RDNA2 lacks hardware FP8 support. Software FP8 emulation is available:
- Stores weights/activations as int8 with per-tensor scale factors
- Dequantizes to FP16 before compute (no speed gain, saves VRAM)
- Uses standard e4m3fn format (not MI300's fnuz)
- Triton-based quantize/dequant kernels are fully portable

### hipGraph Support

PyTorch's `torch.cuda.CUDAGraph()` maps to hipGraph on HIP.
On RDNA2, graph capture requires serialization guards:
- `AMD_SERIALIZE_KERNEL=3` and `AMD_SERIALIZE_COPY=3` during capture
- Prevents concurrent kernel execution that can cause capture failures
- Automatically applied in `cuda_graph_runner.py`

## Known Limitations

1. **No WMMA/matrix cores**: RDNA2 lacks dedicated matrix acceleration
   hardware. All GEMMs run through scalar/vector ALUs via hipBLASLt.
2. **12 GB VRAM**: Consumer RDNA2 cards have limited VRAM. Use
   `--mem-fraction-static 0.35` and smaller models.
3. **Triton attention**: The Triton attention backend is ~7x slower than
   `torch_native` on gfx1030. Always prefer `torch_native`.
4. **Flash Attention**: Requires the Triton backend build
   (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`). CK-based flash-attn
   does not support RDNA2.

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | Maps any gfx103x → gfx1030 ISA |
| `PYTORCH_ROCM_ARCH` | `gfx1030` | PyTorch compile target |
| `AMDGPU_TARGETS` | `gfx1030` | General ROCm compile target |
| `GPU_MAX_HW_QUEUES` | `4` | Prevents HW queue oversubscription |
| `HIP_VISIBLE_DEVICES` | `0` | Limit to single GPU |
| `AMD_SERIALIZE_KERNEL` | `3` | Kernel serialization for graph capture |
| `AMD_SERIALIZE_COPY` | `3` | Copy serialization for graph capture |
| `SGLANG_ROCM_RDNA2` | `1` | Force RDNA2 detection (manual override) |
