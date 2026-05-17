import importlib.util
import pathlib
import sys

import torch
from universal_kv.types import TierKind


def _load_broker_cls():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    broker_path = (
        repo_root
        / "python"
        / "sglang"
        / "srt"
        / "layers"
        / "attention"
        / "universal_kv_broker.py"
    )
    sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("local_universal_kv_broker", broker_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.UniversalKVBroker


def test_allocate_store_materialize():
    UniversalKVBroker = _load_broker_cls()
    broker = UniversalKVBroker(gpu_capacity_mb=512, ram_capacity_mb=4096)
    kv = torch.randn(32, 128)
    block_id = broker.allocate("qwen3.5", layer=0, seq_len=32)
    broker.compress_and_store(block_id, kv, metadata={"importance": 0.9, "bit_width": 3})
    out = broker.materialize_for_model("qwen3.5", layer=0, block_id=block_id)
    assert out.shape == kv.shape
    assert broker.get_record_tier(block_id) == TierKind.VRAM_HOT


def test_cold_block_uses_warm_tier():
    UniversalKVBroker = _load_broker_cls()
    broker = UniversalKVBroker(gpu_capacity_mb=512, ram_capacity_mb=4096)
    kv = torch.randn(16, 64)
    block_id = broker.allocate("lfm2.5", layer=1, seq_len=16)
    broker.compress_and_store(block_id, kv, metadata={"importance": 0.1})
    assert broker.get_record_tier(block_id) == TierKind.RAM_WARM
