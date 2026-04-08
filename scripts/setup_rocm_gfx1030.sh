#!/usr/bin/env bash
# scripts/setup_rocm_gfx1030.sh — Environment setup for RDNA2 (gfx1030) GPUs
# Targets: RX 6700 XT, RX 6800, RX 6900 XT, and all gfx103x variants
# All RDNA2 chips use gfx1030 as the base ISA target for widest compatibility.
set -euo pipefail

echo "=== SGLang ROCm gfx1030 (RDNA2) Environment Setup ==="

# Core compatibility — maps any gfx103x to gfx1030 ISA
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export AMDGPU_TARGETS=gfx1030
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

# GPU stability — prevent runlist oversubscription with SAM/ReBAR
export GPU_MAX_HW_QUEUES=4

# ROCm performance tuning
export HIPBLASLT_TUNING_OVERRIDE_FILE=""
export MIOPEN_FIND_MODE=NORMAL
export MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvDirectNaiveConvFwd

# Triton tuning for Wave32
export TRITON_PRINT_AUTOTUNING=${TRITON_PRINT_AUTOTUNING:-0}

# SGLang defaults for RDNA2
export SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-torch_native}

# Memory: serialization helps with 12GB VRAM contention on RDNA2
export AMD_SERIALIZE_KERNEL=${AMD_SERIALIZE_KERNEL:-3}
export AMD_SERIALIZE_COPY=${AMD_SERIALIZE_COPY:-3}

echo "  HSA_OVERRIDE_GFX_VERSION = $HSA_OVERRIDE_GFX_VERSION"
echo "  PYTORCH_ROCM_ARCH        = $PYTORCH_ROCM_ARCH"
echo "  GPU_MAX_HW_QUEUES        = $GPU_MAX_HW_QUEUES"
echo "  SGLANG_ATTENTION_BACKEND = $SGLANG_ATTENTION_BACKEND"

# Verify ROCm is accessible
if command -v rocminfo &>/dev/null; then
    GPU_NAME=$(rocminfo 2>/dev/null | grep -m1 "Marketing Name" | sed 's/.*: *//')
    GFX_ARCH=$(rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" | sed 's/.*Name: *//')
    echo "  GPU: $GPU_NAME ($GFX_ARCH → gfx1030 via HSA override)"
else
    echo "  WARNING: rocminfo not found — ROCm may not be installed"
fi

# Verify PyTorch+HIP
python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  PyTorch: {torch.__version__}, HIP: {torch.version.hip}')
    print(f'  Device: {props.name} ({props.gcnArchName})')
    print(f'  VRAM: {props.total_mem / 1024**3:.1f} GB')
else:
    print('  WARNING: torch.cuda not available')
" 2>/dev/null || echo "  WARNING: PyTorch not accessible"

echo "=== Environment ready for SGLang on RDNA2 ==="
