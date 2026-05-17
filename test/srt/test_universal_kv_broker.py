import importlib.util
import pathlib
import sys

import pytest
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


def test_capacity_demotes_hot_block_to_warm():
    UniversalKVBroker = _load_broker_cls()
    broker = UniversalKVBroker(gpu_capacity_mb=0, ram_capacity_mb=32)
    kv = torch.randn(64, 64)
    block_id = broker.allocate("qwen3.5", layer=2, seq_len=64)
    broker.compress_and_store(block_id, kv, metadata={"importance": 0.99})
    assert broker.get_record_tier(block_id) == TierKind.RAM_WARM
    metrics = broker.get_metrics()
    assert metrics["gpu_used_bytes"] == 0
    assert metrics["spilled_blocks"] >= 1


def test_capacity_evicts_warm_when_ram_full():
    UniversalKVBroker = _load_broker_cls()
    broker = UniversalKVBroker(gpu_capacity_mb=0, ram_capacity_mb=0)
    kv = torch.randn(32, 32)
    first = broker.allocate("lfm2.5", layer=0, seq_len=32)
    second = broker.allocate("lfm2.5", layer=0, seq_len=32)
    broker.compress_and_store(first, kv, metadata={"importance": 0.1})
    broker.compress_and_store(second, kv, metadata={"importance": 0.2})
    metrics = broker.get_metrics()
    assert metrics["warm_blocks"] == 0
    assert metrics["evicted_blocks"] >= 2


def test_materialize_rejects_model_layer_mismatch():
    UniversalKVBroker = _load_broker_cls()
    broker = UniversalKVBroker(gpu_capacity_mb=64, ram_capacity_mb=64)
    kv = torch.randn(8, 32)
    block_id = broker.allocate("qwen3.5", layer=0, seq_len=8)
    broker.compress_and_store(block_id, kv, metadata={"importance": 1.0})
    with pytest.raises(ValueError):
        broker.materialize_for_model("lfm2.5", layer=0, block_id=block_id)


def test_hot_materialization_uses_header_quant_scale():
    UniversalKVBroker = _load_broker_cls()
    broker = UniversalKVBroker(gpu_capacity_mb=64, ram_capacity_mb=64)
    kv = torch.randn(8, 32)
    block_id = broker.allocate("qwen3.5", layer=0, seq_len=8)
    broker.compress_and_store(
        block_id,
        kv,
        metadata={"importance": 1.0, "bit_width": 8, "scale": 999.0},
    )
    record = broker.records[block_id]
    expected_scale = float(kv.abs().max().clamp(min=1e-8))
    assert record.header.scale == pytest.approx(expected_scale, rel=1e-5)

    materialized = broker.materialize_for_model("qwen3.5", layer=0, block_id=block_id)
    assert torch.allclose(materialized, kv, atol=0.05, rtol=0.05)


def test_rejects_invalid_bit_width():
    UniversalKVBroker = _load_broker_cls()
    broker = UniversalKVBroker(gpu_capacity_mb=64, ram_capacity_mb=64)
    kv = torch.randn(8, 32)
    block_id = broker.allocate("qwen3.5", layer=0, seq_len=8)
    with pytest.raises(ValueError):
        broker.compress_and_store(block_id, kv, metadata={"importance": 1.0, "bit_width": 0})
