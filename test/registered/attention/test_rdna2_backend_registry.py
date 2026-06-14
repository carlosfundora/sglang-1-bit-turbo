"""RDNA2 attention-backend registry / init smoke tests.

CI-friendly (no GPU, no model). Catches the class of regressions seen on gfx1030:
  * a backend module that fails to import (e.g. the triton `UnboundLocalError: torch`
    from a redundant local import — fixed),
  * the `aiter` backend silently routing decode to the slow aiter unified_attention on
    RDNA instead of the fork's fast triton path (fixed: it must resolve to TritonAttnBackend
    on RDNA, since AITER/CK has no RDNA decode kernel).
"""

import importlib

import pytest

from sglang.srt.layers.attention import attention_registry as reg

# Backends that must always be importable + registered on this (ROCm/RDNA2) stack.
RDNA2_BACKENDS = ["triton", "torch_native", "aiter", "atom", "universal_broker", "rdna2_hip"]
# Their backend modules (import-time safety — must not raise on import).
RDNA2_BACKEND_MODULES = [
    "sglang.srt.layers.attention.triton_backend",
    "sglang.srt.layers.attention.torch_native_backend",
    "sglang.srt.layers.attention.aiter_backend",
    "sglang.srt.layers.attention.rdna2_hip_backend",
    "sglang.srt.layers.attention.universal_broker_backend",
]


def test_rdna2_backends_registered():
    for name in RDNA2_BACKENDS:
        assert name in reg.ATTENTION_BACKENDS, f"{name} not registered"
        assert callable(reg.ATTENTION_BACKENDS[name]), f"{name} factory not callable"


@pytest.mark.parametrize("mod", RDNA2_BACKEND_MODULES)
def test_backend_module_imports(mod):
    # Import-time errors (shadowed globals, bad top-level code) surface here.
    importlib.import_module(mod)


def _fake(name):
    cls = type(name, (), {"__init__": lambda self, runner: None})
    return cls


def test_aiter_resolves_to_triton_on_rdna(monkeypatch):
    """On RDNA, --attention-backend aiter must resolve to the triton backend."""
    import sglang.srt.layers.attention.triton_backend as tb
    import sglang.srt.layers.attention.aiter_backend as ab

    FakeTriton, FakeAiter = _fake("FakeTriton"), _fake("FakeAiter")
    monkeypatch.setattr(reg, "_current_gpu_is_rdna", lambda: True)
    monkeypatch.setattr(tb, "TritonAttnBackend", FakeTriton)
    monkeypatch.setattr(ab, "AiterAttnBackend", FakeAiter)

    backend = reg.ATTENTION_BACKENDS["aiter"](object())
    assert isinstance(backend, FakeTriton), "aiter must use the triton path on RDNA"


def test_aiter_uses_aiter_off_rdna(monkeypatch):
    """Off RDNA (CDNA/MI), aiter keeps the real AITER backend."""
    import sglang.srt.layers.attention.triton_backend as tb
    import sglang.srt.layers.attention.aiter_backend as ab

    FakeTriton, FakeAiter = _fake("FakeTriton"), _fake("FakeAiter")
    monkeypatch.setattr(reg, "_current_gpu_is_rdna", lambda: False)
    monkeypatch.setattr(tb, "TritonAttnBackend", FakeTriton)
    monkeypatch.setattr(ab, "AiterAttnBackend", FakeAiter)

    backend = reg.ATTENTION_BACKENDS["aiter"](object())
    assert isinstance(backend, FakeAiter), "aiter must use the AITER backend off RDNA"
