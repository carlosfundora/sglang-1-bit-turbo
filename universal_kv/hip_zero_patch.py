"""
hip_zero_patch.py — Workaround for gfx1031/gfx1030 HSA override fill-kernel crash.

ROOT CAUSE (verified 2026-05-25):
  Physical GPU:           gfx1031 (RX 6700 XT, device 0x73df)
  HSA_OVERRIDE_GFX_VERSION=10.3.0 → forces HSA runtime to report arch as gfx1030
  PyTorch binary:         Compiled ONLY for gfx1031 (torch.cuda.get_arch_list()=['gfx1031'])

  When PyTorch dispatches any of its built-in HIP compute kernels, the dispatch layer
  looks up the pre-compiled kernel for the HSA-reported arch (gfx1030). No gfx1030
  kernel exists in this binary (only gfx1031). Result: null/invalid function pointer →
  SIGSEGV on first kernel launch.

  WHAT WORKS:
    torch.empty()             — just hipMalloc, no kernel
    copy_() / .cuda()         — hipMemcpyHtoD, no kernel
    rocBLAS ops (mm, bmm)     — external library with gfx1030 kernels
    Triton JIT kernels        — JIT-compiled to gfx1030 via AMDGPU_TARGETS=gfx1030
    hipMemset()               — HIP runtime, not PyTorch user kernel
    aiter kernels             — pre-compiled for gfx1030

  WHAT CRASHES:
    torch.zeros/ones/full     — fill kernel (PyTorch built-in)
    Tensor.fill_() / zero_()  — fill kernel
    torch.sum/mean/max        — reduction kernel (PyTorch built-in)
    Elementwise ops (relu, +) — elementwise kernel (PyTorch built-in)

  WHY SERVER INITIALIZES BUT CRASHES AT PREFILL:
    Init = torch.empty + hipMemcpy (weight load) + aiter setup → all safe.
    KV cache = torch.empty for actual K/V buffers → safe.
    Prefill = first torch.zeros(device='cuda') for attention indptr/mask prep →
              SIGSEGV. Then attention compute itself would use Triton/aiter → safe.

PATCHES IN THIS FILE:
    torch.zeros / zeros_like    → hipMemset(0)          — fastest, no kernel
    Tensor.zero_()              → hipMemset(0)
    Tensor.fill_(0)             → hipMemset(0)
    Tensor.fill_(v≠0)           → CPU tensor fill + hipMemcpyHtoD (safe fallback)
    torch.ones / ones_like      → empty + fill_(1) path
    torch.full / full_like      → empty + fill_(value) path
    torch.rand / randn          → CPU generation + .to(device) (hipMemcpy)

    NOTE: torch.sum, elementwise ops (+, relu, etc.) are NOT patched here.
    SGLang's forward path uses aiter/Triton for those — they do not go through
    PyTorch's broken built-in kernels. If you hit an elementwise crash outside
    of aiter/Triton, the only fix is to reinstall PyTorch with gfx1030 support.

USAGE:
    import universal_kv.hip_zero_patch   # first line of server launch script

PERMANENT FIX:
    Reinstall PyTorch with gfx1030 kernels bundled:
      pip install torch --index-url https://download.pytorch.org/whl/rocm<ver>
    Verify: python -c "import torch; print(torch.cuda.get_arch_list())"
    Should include 'gfx1030'. Remove this import once verified.
"""

from __future__ import annotations

import ctypes
import os
import warnings
from typing import Optional

import torch

__all__ = ["apply", "validate", "is_applied"]

_applied = False
_hip_lib: Optional[ctypes.CDLL] = None


# ---------------------------------------------------------------------------
# HIP runtime helpers
# ---------------------------------------------------------------------------

def _get_hip() -> ctypes.CDLL:
    global _hip_lib
    if _hip_lib is not None:
        return _hip_lib
    path = os.environ.get("HIP_ZERO_PATCH_LIB", "/opt/rocm/lib/libamdhip64.so")
    try:
        lib = ctypes.CDLL(path)
    except OSError as e:
        raise RuntimeError(
            f"hip_zero_patch: cannot load {path}. "
            "Set HIP_ZERO_PATCH_LIB env var to correct path."
        ) from e
    lib.hipMemset.restype = ctypes.c_int
    lib.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.hipMemcpy.restype = ctypes.c_int
    lib.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _hip_lib = lib
    return lib


def _memset_zero(t: torch.Tensor) -> None:
    """Zero a contiguous CUDA tensor via hipMemset — bypasses PyTorch fill kernel."""
    if not t.is_contiguous():
        t = t.contiguous()
    rc = _get_hip().hipMemset(
        ctypes.c_void_p(t.data_ptr()),
        ctypes.c_int(0),
        ctypes.c_size_t(t.nbytes),
    )
    if rc != 0:
        raise RuntimeError(f"hipMemset failed: code {rc}")


def _fill_via_cpu(t: torch.Tensor, value) -> None:
    """Fill CUDA tensor via CPU generation + hipMemcpyHtoD. Safe for any dtype/value."""
    cpu_t = torch.empty(t.shape, dtype=t.dtype).fill_(value)
    t.copy_(cpu_t)


# ---------------------------------------------------------------------------
# Patched implementations
# ---------------------------------------------------------------------------

def _patched_fill_(self: torch.Tensor, value) -> torch.Tensor:
    if self.is_cuda and self.data_ptr() != 0:
        if not self.is_contiguous():
            raise RuntimeError(
                "[hip_zero_patch] fill_ on non-contiguous CUDA tensors is unsupported "
                "under gfx1031+gfx1030 override. Call contiguous() first."
            )
        contig = self
        if value == 0:
            _memset_zero(contig)
        else:
            _fill_via_cpu(contig, value)
        return self
    return _orig_fill_(self, value)


def _patched_zero_(self: torch.Tensor) -> torch.Tensor:
    if self.is_cuda and self.data_ptr() != 0:
        if not self.is_contiguous():
            raise RuntimeError(
                "[hip_zero_patch] zero_ on non-contiguous CUDA tensors is unsupported "
                "under gfx1031+gfx1030 override. Call contiguous() first."
            )
        _memset_zero(self)
        return self
    return _orig_zero_(self)


def _patched_zeros(*size, out=None, dtype=None, layout=torch.strided,
                   device=None, requires_grad=False, pin_memory=False, **kwargs):
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = size[0]
    dev_str = str(device) if device is not None else ""
    if dev_str.startswith("cuda") or dev_str.startswith("hip"):
        t = torch.empty(*size, dtype=dtype, layout=layout, device=device,
                        requires_grad=requires_grad)
        _memset_zero(t)
        if out is not None:
            out.copy_(t); return out
        return t
    return _orig_zeros(*size, out=out, dtype=dtype, layout=layout, device=device,
                       requires_grad=requires_grad, pin_memory=pin_memory, **kwargs)


def _patched_zeros_like(input, *, memory_format=None, dtype=None, layout=None,
                        device=None, requires_grad=False, **kwargs):
    tgt_dev = device if device is not None else input.device
    if str(tgt_dev).startswith("cuda") or (hasattr(tgt_dev, 'type') and tgt_dev.type == 'cuda'):
        t = torch.empty_like(input, memory_format=memory_format, dtype=dtype,
                             layout=layout, device=device, requires_grad=requires_grad)
        _memset_zero(t)
        return t
    return _orig_zeros_like(input, memory_format=memory_format, dtype=dtype,
                            layout=layout, device=device, requires_grad=requires_grad)


def _patched_ones(*size, out=None, dtype=None, layout=torch.strided,
                  device=None, requires_grad=False, **kwargs):
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = size[0]
    dev_str = str(device) if device is not None else ""
    if dev_str.startswith("cuda") or dev_str.startswith("hip"):
        t = torch.empty(*size, dtype=dtype, layout=layout, device=device,
                        requires_grad=requires_grad)
        _fill_via_cpu(t, 1)
        if out is not None:
            out.copy_(t); return out
        return t
    return _orig_ones(*size, out=out, dtype=dtype, layout=layout, device=device,
                      requires_grad=requires_grad, **kwargs)


def _patched_ones_like(input, *, memory_format=None, dtype=None, layout=None,
                       device=None, requires_grad=False, **kwargs):
    tgt_dev = device if device is not None else input.device
    if str(tgt_dev).startswith("cuda") or (hasattr(tgt_dev, 'type') and tgt_dev.type == 'cuda'):
        t = torch.empty_like(input, memory_format=memory_format, dtype=dtype,
                             layout=layout, device=device, requires_grad=requires_grad)
        _fill_via_cpu(t, 1)
        return t
    return _orig_ones_like(input, memory_format=memory_format, dtype=dtype,
                           layout=layout, device=device, requires_grad=requires_grad)


def _patched_full(size, fill_value, *, out=None, dtype=None, layout=torch.strided,
                  device=None, requires_grad=False, **kwargs):
    dev_str = str(device) if device is not None else ""
    if dev_str.startswith("cuda") or dev_str.startswith("hip"):
        t = torch.empty(size, dtype=dtype, layout=layout, device=device,
                        requires_grad=requires_grad)
        if fill_value == 0:
            _memset_zero(t)
        else:
            _fill_via_cpu(t, fill_value)
        if out is not None:
            out.copy_(t); return out
        return t
    return _orig_full(size, fill_value, out=out, dtype=dtype, layout=layout,
                      device=device, requires_grad=requires_grad, **kwargs)


def _patched_full_like(input, fill_value, *, memory_format=None, dtype=None,
                       layout=None, device=None, requires_grad=False, **kwargs):
    tgt_dev = device if device is not None else input.device
    if str(tgt_dev).startswith("cuda") or (hasattr(tgt_dev, 'type') and tgt_dev.type == 'cuda'):
        t = torch.empty_like(input, memory_format=memory_format, dtype=dtype,
                             layout=layout, device=device, requires_grad=requires_grad)
        if fill_value == 0:
            _memset_zero(t)
        else:
            _fill_via_cpu(t, fill_value)
        return t
    return _orig_full_like(input, fill_value, memory_format=memory_format,
                           dtype=dtype, layout=layout, device=device,
                           requires_grad=requires_grad)


def _patched_rand(*size, generator=None, out=None, dtype=None, layout=torch.strided,
                  device=None, requires_grad=False, **kwargs):
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = size[0]
    dev_str = str(device) if device is not None else ""
    if dev_str.startswith("cuda") or dev_str.startswith("hip"):
        cpu_t = _orig_rand(*size, generator=generator, dtype=dtype or torch.float32,
                           layout=layout, requires_grad=False)
        return cpu_t.to(device=device, non_blocking=False).requires_grad_(requires_grad)
    return _orig_rand(*size, generator=generator, out=out, dtype=dtype,
                      layout=layout, device=device, requires_grad=requires_grad)


def _patched_randn(*size, generator=None, out=None, dtype=None, layout=torch.strided,
                   device=None, requires_grad=False, **kwargs):
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = size[0]
    dev_str = str(device) if device is not None else ""
    if dev_str.startswith("cuda") or dev_str.startswith("hip"):
        cpu_t = _orig_randn(*size, generator=generator, dtype=dtype or torch.float32,
                            layout=layout, requires_grad=False)
        return cpu_t.to(device=device, non_blocking=False).requires_grad_(requires_grad)
    return _orig_randn(*size, generator=generator, out=out, dtype=dtype,
                       layout=layout, device=device, requires_grad=requires_grad)


# Save originals at module-load time (no CUDA ops)
_orig_fill_ = torch.Tensor.fill_
_orig_zero_ = torch.Tensor.zero_
_orig_zeros = torch.zeros
_orig_zeros_like = torch.zeros_like
_orig_ones = torch.ones
_orig_ones_like = torch.ones_like
_orig_full = torch.full
_orig_full_like = torch.full_like
_orig_rand = torch.rand
_orig_randn = torch.randn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_applied() -> bool:
    return _applied


def apply() -> None:
    """Apply all patches. Idempotent."""
    global _applied
    if _applied:
        return
    torch.Tensor.fill_ = _patched_fill_
    torch.Tensor.zero_ = _patched_zero_
    torch.zeros = _patched_zeros
    torch.zeros_like = _patched_zeros_like
    torch.ones = _patched_ones
    torch.ones_like = _patched_ones_like
    torch.full = _patched_full
    torch.full_like = _patched_full_like
    torch.rand = _patched_rand
    torch.randn = _patched_randn
    _applied = True
    warnings.warn(
        "[hip_zero_patch] fill-kernel workaround active (gfx1031+gfx1030 override). "
        "Permanent fix: reinstall PyTorch with gfx1030 kernel support.",
        UserWarning, stacklevel=2,
    )


def _read_gpu_bytes(t: torch.Tensor, nbytes: int) -> ctypes.Array:
    """Read nbytes from the start of a GPU tensor via hipMemcpy. No PyTorch kernel."""
    hip = _get_hip()
    buf = (ctypes.c_uint8 * nbytes)()
    hip.hipMemcpy(
        ctypes.c_void_p(ctypes.addressof(buf)),
        ctypes.c_void_p(t.data_ptr()),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(2),  # hipMemcpyDeviceToHost
    )
    return buf


def validate() -> bool:
    """Smoke-test patched ops. Reads results via hipMemcpy to avoid broken reduction kernels."""
    if not _applied:
        apply()
    try:
        import struct

        def _read_f32(t, idx=0):
            raw = _read_gpu_bytes(t, (idx + 1) * 4)
            return struct.unpack_from('<f', bytes(raw), idx * 4)[0]

        def _read_i32(t, idx=0):
            raw = _read_gpu_bytes(t, (idx + 1) * 4)
            return struct.unpack_from('<i', bytes(raw), idx * 4)[0]

        # zeros
        x = torch.zeros(32, device="cuda", dtype=torch.float32)
        assert _read_f32(x, 0) == 0.0 and _read_f32(x, 16) == 0.0, "zeros failed"

        # zeros_like
        y = torch.zeros_like(x)
        assert _read_f32(y, 0) == 0.0, "zeros_like failed"

        # zero_()
        x.zero_()
        assert _read_f32(x, 0) == 0.0, "zero_() failed"

        # fill_(0)
        x.fill_(0.0)
        assert _read_f32(x, 0) == 0.0, "fill_(0) failed"

        # fill_(3.0)
        x.fill_(3.0)
        v = _read_f32(x, 0)
        assert abs(v - 3.0) < 0.01, f"fill_(3.0) got {v}"

        # ones
        o = torch.ones(32, device="cuda", dtype=torch.float32)
        assert abs(_read_f32(o, 0) - 1.0) < 0.01, "ones failed"

        # ones_like
        o2 = torch.ones_like(x)
        assert abs(_read_f32(o2, 0) - 1.0) < 0.01, "ones_like failed"

        # full with negative float
        f = torch.full((32,), -1.0, device="cuda", dtype=torch.float32)
        assert abs(_read_f32(f, 0) - (-1.0)) < 0.01, "full(-1.0) failed"

        # full int32 — SGLang block_kv_indices pattern
        fi = torch.full((8,), -1, device="cuda", dtype=torch.int32)
        assert _read_i32(fi, 0) == -1, f"full int32 got {_read_i32(fi, 0)}"

        # int32 zeros — SGLang kv_indptr / extend_attention pattern
        indptr = torch.zeros(8, dtype=torch.int32, device="cuda")
        assert _read_i32(indptr, 0) == 0, f"int32 zeros got {_read_i32(indptr, 0)}"

        # window_start_pos pattern (extend_attention.py:1114)
        ws = torch.zeros(4, dtype=torch.int32, device="cuda")
        assert _read_i32(ws, 0) == 0, "window_start_pos zeros failed"

        # fp16 zeros (kv cache mask pattern)
        fp16z = torch.zeros(64, dtype=torch.float16, device="cuda")
        raw = _read_gpu_bytes(fp16z, 2)
        assert raw[0] == 0 and raw[1] == 0, "fp16 zeros failed"

        # GEMM must still work (rocBLAS — external, unaffected)
        a = torch.ones(16, 16, device="cuda", dtype=torch.float16)
        b = torch.ones(16, 16, device="cuda", dtype=torch.float16)
        c = torch.mm(a, b)
        torch.cuda.synchronize()

        return True
    except Exception as e:
        warnings.warn(f"[hip_zero_patch] validate FAILED: {e}")
        import traceback; traceback.print_exc()
        return False


# Intentionally not auto-applying on import.
# Activation is controlled by callers (e.g. sglang.__init__) for deterministic startup behavior.


# ---------------------------------------------------------------------------
# Self-gating auto-detect
# ---------------------------------------------------------------------------
#
# The bug only manifests when the running PyTorch binary has NO compiled kernel
# for the arch the HSA runtime *reports*. On this stack the physical GPU is
# gfx1031 reported as gfx1030 (HSA_OVERRIDE_GFX_VERSION=10.3.0), but torch is
# built FOR gfx1030 (get_arch_list()==['gfx1030']) — so the reported arch IS in
# the kernel set and fills work natively. `needed()` returns False there and the
# patch stays dormant; it only activates if torch is ever rebuilt without a
# gfx1030 kernel (e.g. a gfx1031-only wheel under the same override).


def _override_to_arch(override: str) -> str:
    """'10.3.0' -> 'gfx1030'."""
    parts = override.split(".")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return f"gfx{parts[0]}{parts[1]}{parts[2]}"
    return ""


def needed() -> bool:
    """True iff the running torch lacks a kernel for the HSA-reported arch.

    This is the exact fill-kernel-segfault condition: gfx1031-only torch under an
    HSA override to gfx1030 (reported arch not in get_arch_list()). Returns False
    (patch unnecessary) when torch is built for the reported arch — the normal case
    on this box — so wiring it into startup is safe and a no-op when not required.
    """
    try:
        if torch.version.hip is None or not torch.cuda.is_available():
            return False
        arch_list = [a.split(":")[0] for a in torch.cuda.get_arch_list()]
        try:
            reported = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        except Exception:
            reported = ""
        target = reported or _override_to_arch(
            os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
        )
        if not target:
            return False
        return target not in arch_list
    except Exception:
        return False


def auto_apply_if_needed() -> bool:
    """Apply the patch only if `needed()` (or SGLANG_FORCE_HIP_ZERO_PATCH=1). No-op otherwise.

    Returns True if the patch was applied. Safe to call unconditionally at startup.
    """
    if _applied:
        return True
    if os.environ.get("SGLANG_FORCE_HIP_ZERO_PATCH") == "1" or needed():
        apply()
        return True
    return False
