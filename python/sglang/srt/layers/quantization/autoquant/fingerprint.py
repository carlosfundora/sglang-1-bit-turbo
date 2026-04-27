"""AutoQuant fingerprint — identifies (gpu, model) pairs for policy reuse.

Modelled after sglang/srt/speculative/sok/fingerprint.py — a frozen
dataclass whose hex digest keys the on-disk policy shelf.

Any change to a fingerprint field forces a fresh learn run. Policies
are NEVER reused across mismatched fingerprints.
"""

import hashlib
import json
import logging
import os
import platform
import sys
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoQuantFingerprint:
    """Identity for an (environment, model) pair."""

    # Hardware
    gpu_arch: str           # e.g. "gfx1030"
    wave_size: int          # 32 or 64

    # Software stack (codec correctness depends on triton + rocm)
    rocm_version: str
    triton_version: str
    python_version: str

    # Model structure (codec choice depends on these)
    model_family: str       # "qwen3-4b", "lfm2-2.6b", ...
    n_layers: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    dtype_mode: str         # "fp16", "bf16"

    # AutoQuant codec set version — bump when codec implementations change
    codec_set_version: str

    @property
    def hex_digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "AutoQuantFingerprint":
        return cls(**json.loads(s))


def _detect_gpu():
    """Returns (gpu_arch, wave_size, rocm_version) — best-effort, never raises."""
    gpu_arch = "unknown"
    wave_size = 32
    rocm_version = "unknown"
    try:
        import torch
        if hasattr(torch.version, "hip") and torch.version.hip:
            rocm_version = torch.version.hip
            try:
                props = torch.cuda.get_device_properties(0)
                gcn = getattr(props, "gcnArchName", "")
                if gcn:
                    # "gfx1030:sramecc+:xnack-" → "gfx1030"
                    gpu_arch = gcn.split(":")[0]
                wave_size = 32 if ("gfx10" in gpu_arch or "gfx11" in gpu_arch) else 64
            except Exception as e:
                logger.debug("AutoQuant: could not query gpu props: %s", e)
        elif torch.cuda.is_available():
            # NVIDIA path — derive arch from compute capability.
            try:
                cc = torch.cuda.get_device_capability(0)
                gpu_arch = f"sm_{cc[0]}{cc[1]}"
                wave_size = 32  # NVIDIA warps
            except Exception:
                pass
    except Exception as e:
        logger.debug("AutoQuant: torch import or cuda probe failed: %s", e)
    return gpu_arch, wave_size, rocm_version


def _detect_triton_version() -> str:
    try:
        import triton
        return getattr(triton, "__version__", "unknown")
    except ImportError:
        return "none"


def detect_fingerprint(
    *,
    model_family: str,
    n_layers: int,
    head_dim: int,
    n_heads: int,
    n_kv_heads: int,
    dtype_mode: str = "fp16",
    codec_set_version: str = "1",
) -> AutoQuantFingerprint:
    """Build an AutoQuantFingerprint from runtime state + caller-supplied
    model metadata.

    Caller (the KV pool) knows model_family/n_layers/head_dim already; this
    function only auto-detects environment fields.
    """
    gpu_arch, wave_size, rocm_version = _detect_gpu()
    triton_version = _detect_triton_version()
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    fp = AutoQuantFingerprint(
        gpu_arch=gpu_arch,
        wave_size=wave_size,
        rocm_version=rocm_version,
        triton_version=triton_version,
        python_version=python_version,
        model_family=model_family,
        n_layers=n_layers,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dtype_mode=dtype_mode,
        codec_set_version=codec_set_version,
    )
    logger.info(
        "AutoQuant: fingerprint %s "
        "(model=%s, gpu=%s, rocm=%s, triton=%s, layers=%d, hd=%d)",
        fp.hex_digest, model_family, gpu_arch, rocm_version,
        triton_version, n_layers, head_dim,
    )
    return fp
