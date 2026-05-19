import importlib.util
import pathlib

import torch


def _load_spill_cls():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    spill_path = (
        repo_root / "python" / "sglang" / "srt" / "mem_cache" / "universal_kv_spill.py"
    )
    spec = importlib.util.spec_from_file_location("local_universal_kv_spill", spill_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.UniversalKVSpillManager


def test_spill_and_restore_roundtrip():
    UniversalKVSpillManager = _load_spill_cls()
    mgr = UniversalKVSpillManager(pin_memory=True)
    x = torch.randn(128, 128, dtype=torch.float16)
    key = mgr.spill("req-1", x)
    y = mgr.restore(key)
    assert y.shape == x.shape
    assert y.device.type == "cpu"


def test_spill_nested_payload_and_discard():
    UniversalKVSpillManager = _load_spill_cls()
    mgr = UniversalKVSpillManager(pin_memory=False)
    payload = {
        "compressed_hot": torch.randint(0, 8, (8, 8), dtype=torch.uint8),
        "residual_warm": torch.ones(8, 8, dtype=torch.int8),
    }
    key = mgr.spill("req-2", payload)
    restored = mgr.restore(key)
    assert restored["compressed_hot"].device.type == "cpu"
    assert restored["residual_warm"].device.type == "cpu"
    mgr.discard(key)
    assert key not in mgr.store
