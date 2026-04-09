"""RDNA2 kernel dispatch — routes SGLang layer ops to Wave32 HIP kernels.

This module provides a unified interface for all RDNA2-optimized operations.
It probes kernel availability at import time and provides clean fallback
to existing Triton/PyTorch paths.

Usage in SGLang layers:
    from sglang.srt.layers.kernels.rdna2.dispatch import rdna2_ops
    if rdna2_ops.available:
        rdna2_ops.rms_norm(out, input, weight, eps)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class RDNA2Ops:
    """Registry of available RDNA2-optimized operations.

    Each operation lazily compiles on first use and caches the result.
    If compilation fails, the operation returns None and callers should
    fall back to their default implementation.
    """

    available: bool = False
    _probed: bool = False
    _compilation_errors: list = field(default_factory=list)

    def probe(self) -> bool:
        """Check if RDNA2 HIP kernels can be compiled on this system."""
        if self._probed:
            return self.available

        self._probed = True

        # Must be on ROCm
        if not torch.cuda.is_available() or not hasattr(torch.version, "hip"):
            logger.debug("RDNA2 kernels: not on ROCm, disabled")
            return False

        # Must have hipcc
        import shutil

        if not shutil.which("hipcc"):
            logger.debug("RDNA2 kernels: hipcc not found, disabled")
            return False

        # Must be RDNA2 (or forced)
        force = os.environ.get("SGLANG_RDNA2_KERNELS", "").lower()
        if force == "0" or force == "false":
            logger.info("RDNA2 kernels: disabled by SGLANG_RDNA2_KERNELS=0")
            return False
        if force == "1" or force == "true":
            self.available = True
            logger.info("RDNA2 kernels: force-enabled by SGLANG_RDNA2_KERNELS=1")
            return True

        try:
            from sglang.srt.hardware_backend.rocm.arch_detection import is_rdna2

            self.available = is_rdna2()
        except ImportError:
            # Fallback: check env
            self.available = os.environ.get(
                "HSA_OVERRIDE_GFX_VERSION", ""
            ).startswith("10.3")

        if self.available:
            logger.info("RDNA2 kernels: enabled (gfx1030 detected)")
        else:
            logger.debug("RDNA2 kernels: not RDNA2 hardware, disabled")

        return self.available

    # ──── RMSNorm ────

    def rms_norm(
        self, out: Tensor, input: Tensor, weight: Tensor, epsilon: float = 1e-6
    ) -> Optional[Tensor]:
        """Wave32 RMSNorm. Returns out on success, None on fallback."""
        if not self.probe():
            return None
        try:
            from .rmsnorm import rms_norm

            return rms_norm(out, input, weight, epsilon)
        except Exception as e:
            self._compilation_errors.append(("rms_norm", str(e)))
            logger.debug(f"RDNA2 RMSNorm fallback: {e}")
            return None

    def fused_add_rms_norm(
        self,
        input: Tensor,
        residual: Tensor,
        weight: Tensor,
        epsilon: float = 1e-6,
    ) -> Optional[tuple]:
        """Wave32 fused add+RMSNorm. Returns (input, residual) or None."""
        if not self.probe():
            return None
        try:
            from .rmsnorm import fused_add_rms_norm

            return fused_add_rms_norm(input, residual, weight, epsilon)
        except Exception as e:
            self._compilation_errors.append(("fused_add_rms_norm", str(e)))
            logger.debug(f"RDNA2 fused_add_rms_norm fallback: {e}")
            return None

    # ──── FP8 Dequant ────

    def fp8_dequantize(
        self,
        input: Tensor,
        scale: float,
        output_dtype: torch.dtype = torch.float16,
        block_size: Optional[int] = None,
        block_scales: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Wave32 FP8 dequant. Returns tensor or None."""
        if not self.probe():
            return None
        try:
            from .fp8_dequant import fp8_dequantize

            return fp8_dequantize(input, scale, output_dtype, block_size, block_scales)
        except Exception as e:
            self._compilation_errors.append(("fp8_dequant", str(e)))
            logger.debug(f"RDNA2 FP8 dequant fallback: {e}")
            return None

    def fp8_quantize(self, input: Tensor, scale: float) -> Optional[Tensor]:
        """Wave32 FP8 quantize. Returns uint8 tensor or None."""
        if not self.probe():
            return None
        try:
            from .fp8_dequant import fp8_quantize

            return fp8_quantize(input, scale)
        except Exception as e:
            self._compilation_errors.append(("fp8_quant", str(e)))
            logger.debug(f"RDNA2 FP8 quant fallback: {e}")
            return None

    # ──── RoPE ────

    def apply_rotary_pos_emb(
        self,
        query: Tensor,
        key: Tensor,
        cos_cache: Tensor,
        sin_cache: Tensor,
        positions: Tensor,
        head_size: int,
        num_heads: int,
        num_kv_heads: int,
        rot_dim: int,
    ) -> Optional[tuple]:
        """Wave32 RoPE. Returns (query, key) or None."""
        if not self.probe():
            return None
        try:
            from .rope import apply_rotary_pos_emb_neox

            return apply_rotary_pos_emb_neox(
                query,
                key,
                cos_cache,
                sin_cache,
                positions,
                head_size,
                num_heads,
                num_kv_heads,
                rot_dim,
            )
        except Exception as e:
            self._compilation_errors.append(("rope", str(e)))
            logger.debug(f"RDNA2 RoPE fallback: {e}")
            return None

    # ──── Activations ────

    def silu_and_mul(self, input: Tensor) -> Optional[Tensor]:
        """Wave32 fused SiLU-gate. Returns output or None."""
        if not self.probe():
            return None
        try:
            from .activations import silu_and_mul

            return silu_and_mul(input)
        except Exception as e:
            self._compilation_errors.append(("silu_and_mul", str(e)))
            logger.debug(f"RDNA2 SiLU-gate fallback: {e}")
            return None

    def gelu_and_mul(self, input: Tensor) -> Optional[Tensor]:
        """Wave32 fused GELU-gate. Returns output or None."""
        if not self.probe():
            return None
        try:
            from .activations import gelu_and_mul

            return gelu_and_mul(input)
        except Exception as e:
            self._compilation_errors.append(("gelu_and_mul", str(e)))
            logger.debug(f"RDNA2 GELU-gate fallback: {e}")
            return None

    # ──── Diagnostics ────

    def status(self) -> dict:
        """Return diagnostic info about kernel availability."""
        self.probe()
        return {
            "available": self.available,
            "backend": "hip_wave32" if self.available else "fallback",
            "compilation_errors": list(self._compilation_errors),
            "kernels": {
                "rmsnorm": self._check_kernel("rmsnorm"),
                "fp8_dequant": self._check_kernel("fp8_dequant"),
                "rope": self._check_kernel("rope"),
                "activations": self._check_kernel("activations"),
            },
        }

    def _check_kernel(self, name: str) -> str:
        """Check if a specific kernel module can be imported."""
        if not self.available:
            return "disabled"
        try:
            mod = __import__(
                f"sglang.srt.layers.kernels.rdna2.{name}", fromlist=[name]
            )
            get_fn = getattr(mod, "_get_module", None)
            if get_fn:
                return "compiled" if get_fn() is not None else "compilation_pending"
            return "available"
        except ImportError:
            return "missing"


# Singleton instance — import this in SGLang layers
rdna2_ops = RDNA2Ops()
