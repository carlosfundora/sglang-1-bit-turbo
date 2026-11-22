#!/usr/bin/env python3
"""SGLang RDNA2 (gfx1030) compatibility check.

Run this script to validate that your ROCm environment is correctly
configured for running SGLang on RDNA2 GPUs (RX 6000 series).

Usage:
    python scripts/check_rocm_gfx1030.py
"""

import os
import sys
import subprocess
import json


def check_env_vars():
    """Check required environment variables."""
    print("\n=== Environment Variables ===")
    required = {
        "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
    }
    recommended = {
        "PYTORCH_ROCM_ARCH": "gfx1030",
        "GPU_MAX_HW_QUEUES": "4",
        "HIP_VISIBLE_DEVICES": "0",
    }

    ok = True
    for var, expected in required.items():
        val = os.environ.get(var, "")
        status = "✅" if val == expected else "❌"
        if val != expected:
            ok = False
        print(f"  {status} {var} = {val!r} (expected: {expected!r})")

    for var, expected in recommended.items():
        val = os.environ.get(var, "")
        status = "✅" if val == expected else "⚠️"
        print(f"  {status} {var} = {val!r} (recommended: {expected!r})")

    return ok


def check_rocm():
    """Check ROCm installation."""
    print("\n=== ROCm Installation ===")
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.split("\n")
        gpu_name = next((l for l in lines if "Marketing Name" in l), "Unknown")
        gfx_arch = next((l for l in lines if "Name:" in l and "gfx" in l), "Unknown")
        print(f"  ✅ rocminfo available")
        print(f"    GPU: {gpu_name.strip()}")
        print(f"    Arch: {gfx_arch.strip()}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ❌ rocminfo not found or timed out")
        return False


def check_pytorch():
    """Check PyTorch + HIP."""
    print("\n=== PyTorch + HIP ===")
    try:
        import torch

        if not torch.cuda.is_available():
            print("  ❌ torch.cuda not available (HIP not detected)")
            return False

        props = torch.cuda.get_device_properties(0)
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"    HIP version: {torch.version.hip}")
        print(f"    Device: {props.name}")
        print(f"    GCN arch: {props.gcnArchName}")
        total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f"    VRAM: {total_mem / 1024**3:.1f} GB")
        print(f"    Multiprocessors (CUs): {props.multi_processor_count}")

        # Check if it's RDNA2
        if "gfx103" in props.gcnArchName:
            print(f"  ✅ RDNA2 GPU detected — gfx1030 target applies")
        elif "gfx110" in props.gcnArchName:
            print(f"  ⚠️ RDNA3 GPU detected — gfx1030 configs may not be optimal")
        elif "gfx94" in props.gcnArchName:
            print(f"  ℹ️ MI300 GPU detected — use default CDNA configs instead")
        else:
            print(f"  ⚠️ Unknown arch — gfx1030 compatibility not verified")

        return True
    except ImportError:
        print("  ❌ PyTorch not installed")
        return False


def check_triton():
    """Check Triton availability."""
    print("\n=== Triton ===")
    try:
        import triton

        print(f"  ✅ Triton {triton.__version__}")

        # Check for AMD backend
        try:
            from triton.backends.amd import HIPBackend  # noqa: F401

            print(f"  ✅ AMD/HIP backend available")
        except ImportError:
            try:
                from triton.runtime.driver import HIPDriver  # noqa: F401

                print(f"  ✅ HIP driver available")
            except ImportError:
                print(f"  ⚠️ AMD backend not confirmed (may work via default)")

        return True
    except ImportError:
        print("  ❌ Triton not installed")
        return False


def check_sglang():
    """Check SGLang and RDNA2 backend."""
    print("\n=== SGLang ===")
    try:
        import sglang

        print(f"  ✅ SGLang available")

        try:
            from sglang.srt.hardware_backend.rocm import (
                get_rocm_arch,
                is_rdna2,
                get_wave_size,
                is_fp8_hw_available,
            )

            arch = get_rocm_arch()
            rdna2 = is_rdna2()
            wave = get_wave_size()
            fp8 = is_fp8_hw_available()
            print(f"  ✅ ROCm hardware backend available")
            print(f"    Detected arch: {arch}")
            print(f"    RDNA2: {rdna2}")
            print(f"    Wave size: {wave}")
            print(f"    FP8 hardware: {fp8}")
        except ImportError:
            print("  ⚠️ ROCm hardware backend not available (install sglang-1-bit-turbo fork)")

        try:
            from sglang.srt.hardware_backend.rocm.te_integration import (
                has_transformer_engine,
            )

            if has_transformer_engine():
                print(f"  ✅ TransformerEngine integration available")
            else:
                print(f"  ℹ️ TransformerEngine not installed (optional)")
        except ImportError:
            pass

        return True
    except ImportError:
        print("  ❌ SGLang not installed")
        return False


def check_flash_attention():
    """Check flash-attention availability."""
    print("\n=== Flash Attention ===")
    try:
        import flash_attn

        print(f"  ✅ flash-attn {flash_attn.__version__}")
        return True
    except ImportError:
        print("  ℹ️ flash-attn not installed (optional, use triton or torch_native backend)")
        return False


def main():
    print("=" * 60)
    print("  SGLang RDNA2 (gfx1030) Compatibility Check")
    print("=" * 60)

    results = {}
    results["env"] = check_env_vars()
    results["rocm"] = check_rocm()
    results["pytorch"] = check_pytorch()
    results["triton"] = check_triton()
    results["sglang"] = check_sglang()
    results["flash_attn"] = check_flash_attention()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    critical = ["rocm", "pytorch"]
    all_critical = all(results.get(k, False) for k in critical)

    for name, ok in results.items():
        status = "✅ PASS" if ok else ("❌ FAIL" if name in critical else "⚠️ WARN")
        print(f"  {status}: {name}")

    if all_critical:
        print("\n  🚀 System is ready for SGLang on RDNA2!")
        if not results["env"]:
            print("  ⚠️ Run 'source scripts/setup_rocm_gfx1030.sh' to set env vars")
    else:
        print("\n  ❌ Critical issues detected — fix before running SGLang")

    return 0 if all_critical else 1


if __name__ == "__main__":
    sys.exit(main())
