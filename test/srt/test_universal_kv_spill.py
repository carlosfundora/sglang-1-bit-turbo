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
