# Copyright 2026 SGLang-1-bit-turbo Team
# Licensed under the Apache License, Version 2.0
"""ROCm hardware backend for SGLang — RDNA2 (gfx1030) optimized defaults."""

from sglang.srt.hardware_backend.rocm.arch_detection import (
    get_rocm_arch,
    get_wave_size,
    is_rdna2,
    is_rdna3,
    is_cdna,
    is_fp8_hw_available,
)
from sglang.srt.hardware_backend.rocm.gfx1031_defaults import (
    get_gfx1031_server_defaults,
    apply_gfx1031_env,
)
