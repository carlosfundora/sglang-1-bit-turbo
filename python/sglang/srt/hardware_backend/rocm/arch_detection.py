# Copyright 2026 SGLang-1-bit-turbo Team
# Licensed under the Apache License, Version 2.0
"""AMD GPU architecture detection utilities for ROCm."""

import functools
import logging
import os
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Architecture families
_RDNA2_CHIPS = {"gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"}
_RDNA3_CHIPS = {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151"}
_CDNA2_CHIPS = {"gfx90a"}
_CDNA3_CHIPS = {"gfx940", "gfx941", "gfx942"}
_CDNA3X_CHIPS = {"gfx950"}

# Wave size by architecture family
_WAVE32_FAMILIES = _RDNA2_CHIPS | _RDNA3_CHIPS
_WAVE64_FAMILIES = _CDNA2_CHIPS | _CDNA3_CHIPS | _CDNA3X_CHIPS

# FP8 hardware support (native matrix FP8)
_FP8_HW_CHIPS = _CDNA3_CHIPS | _CDNA3X_CHIPS

# gfx1031 compatibility: maps to gfx1030 for code generation
_COMPAT_MAP = {
    "gfx1031": "gfx1030",
    "gfx1032": "gfx1030",
    "gfx1033": "gfx1030",
    "gfx1034": "gfx1030",
    "gfx1035": "gfx1030",
    "gfx1036": "gfx1030",
}


@functools.lru_cache(maxsize=1)
def get_rocm_arch() -> Optional[str]:
    """Detect the GPU architecture string (e.g., 'gfx1031').

    Tries multiple detection methods:
    1. PYTORCH_ROCM_ARCH environment variable
    2. rocminfo agent parsing
    3. PyTorch device properties
    """
    # Method 1: Environment variable
    env_arch = os.environ.get("PYTORCH_ROCM_ARCH", "").strip()
    if env_arch:
        arch = env_arch.split(";")[0].split(",")[0].strip().lower()
        if arch.startswith("gfx"):
            logger.info(f"ROCm arch from PYTORCH_ROCM_ARCH: {arch}")
            return arch

    # Method 2: PyTorch device properties
    try:
        import torch
        if torch.cuda.is_available() and hasattr(torch.version, "hip"):
            props = torch.cuda.get_device_properties(0)
            gcn_arch = getattr(props, "gcnArchName", None)
            if gcn_arch:
                arch = gcn_arch.split(":")[0].strip().lower()
                logger.info(f"ROCm arch from PyTorch: {arch}")
                return arch
    except Exception:
        pass

    # Method 3: rocminfo
    try:
        import subprocess
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "gfx" in line.lower() and "name:" in line.lower():
                match = re.search(r"(gfx\d+[a-z]*)", line.lower())
                if match:
                    arch = match.group(1)
                    logger.info(f"ROCm arch from rocminfo: {arch}")
                    return arch
    except Exception:
        pass

    logger.warning("Could not detect ROCm GPU architecture")
    return None


@functools.lru_cache(maxsize=1)
def get_wave_size() -> int:
    """Return the wavefront size for the detected architecture."""
    arch = get_rocm_arch()
    if arch and arch in _WAVE32_FAMILIES:
        return 32
    return 64


@functools.lru_cache(maxsize=1)
def get_compat_arch() -> Optional[str]:
    """Return the compatibility target architecture for code generation.

    For gfx1031, returns gfx1030 (the base RDNA2 target that compilers support).
    """
    arch = get_rocm_arch()
    if arch is None:
        return None
    return _COMPAT_MAP.get(arch, arch)


def is_rdna2() -> bool:
    """Check if the GPU is RDNA2 architecture."""
    arch = get_rocm_arch()
    return arch is not None and arch in _RDNA2_CHIPS


def is_rdna3() -> bool:
    """Check if the GPU is RDNA3 architecture."""
    arch = get_rocm_arch()
    return arch is not None and arch in _RDNA3_CHIPS


def is_cdna() -> bool:
    """Check if the GPU is CDNA architecture (MI-series)."""
    arch = get_rocm_arch()
    return arch is not None and arch in (_CDNA2_CHIPS | _CDNA3_CHIPS | _CDNA3X_CHIPS)


def is_fp8_hw_available() -> bool:
    """Check if hardware FP8 matrix multiply is available."""
    arch = get_rocm_arch()
    return arch is not None and arch in _FP8_HW_CHIPS


def get_compute_units() -> Optional[int]:
    """Get the number of compute units on the GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        pass
    return None


def get_vram_gb() -> Optional[float]:
    """Get total VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    except Exception:
        pass
    return None


def get_arch_summary() -> dict:
    """Return a summary of the detected GPU architecture and capabilities."""
    arch = get_rocm_arch()
    return {
        "arch": arch,
        "compat_arch": get_compat_arch(),
        "wave_size": get_wave_size(),
        "family": (
            "RDNA2" if is_rdna2() else
            "RDNA3" if is_rdna3() else
            "CDNA" if is_cdna() else
            "unknown"
        ),
        "fp8_hw": is_fp8_hw_available(),
        "compute_units": get_compute_units(),
        "vram_gb": get_vram_gb(),
    }
