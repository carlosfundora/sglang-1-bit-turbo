from universal_kv.model_registry import ModelShapeRegistry
from universal_kv.types import UniversalKVBlockHeader

import pytest


def test_block_header_roundtrip():
    header = UniversalKVBlockHeader(
        block_size=16,
        bit_width=3,
        rotor_id=7,
        origin_model_tag=2,
        turbo_residual_flag=True,
        scale=0.125,
    )
    payload = header.to_bytes()
    restored = UniversalKVBlockHeader.from_bytes(payload)
    assert restored.block_size == 16
    assert restored.bit_width == 3
    assert restored.rotor_id == 7
    assert restored.origin_model_tag == 2
    assert restored.turbo_residual_flag is True


def test_model_shape_registry_register_lookup():
    registry = ModelShapeRegistry()
    registry.register("qwen3.5", num_layers=36, num_kv_heads=8, head_dim=128)
    entry = registry.lookup("qwen3.5")
    assert entry.num_layers == 36
    assert entry.num_kv_heads == 8
    assert entry.head_dim == 128


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_size": 0, "bit_width": 3, "rotor_id": 7, "origin_model_tag": 2, "turbo_residual_flag": True, "scale": 0.125},
        {"block_size": 16, "bit_width": 0, "rotor_id": 7, "origin_model_tag": 2, "turbo_residual_flag": True, "scale": 0.125},
        {"block_size": 16, "bit_width": 3, "rotor_id": -1, "origin_model_tag": 2, "turbo_residual_flag": True, "scale": 0.125},
        {"block_size": 16, "bit_width": 3, "rotor_id": 7, "origin_model_tag": 999, "turbo_residual_flag": True, "scale": 0.125},
        {"block_size": 16, "bit_width": 3, "rotor_id": 7, "origin_model_tag": 2, "turbo_residual_flag": True, "scale": 0.0},
    ],
)
def test_block_header_rejects_invalid_fields(kwargs):
    with pytest.raises(ValueError):
        UniversalKVBlockHeader(**kwargs)
