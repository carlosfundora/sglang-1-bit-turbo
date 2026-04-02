from __future__ import annotations

from pathlib import Path

import gguf
import gguf.quants as gguf_quants
import numpy as np
import pytest

from sglang.srt.utils.gguf_compat import (
    PRISM_Q1_0,
    PRISM_Q1_0_G128,
    ensure_prism_gguf_compat,
)


def _encode_prism_block(scale: float, signs: list[int], block_size: int) -> np.ndarray:
    assert len(signs) == block_size
    packed = np.packbits(np.array(signs, dtype=np.uint8), bitorder="little")
    scale_bytes = np.array([scale], dtype=np.float16).view(np.uint8)
    return np.concatenate([scale_bytes, packed], axis=0)


def test_prism_gguf_compat_patches_quant_sizes_and_dequant():
    ensure_prism_gguf_compat()

    assert gguf.GGML_QUANT_SIZES[PRISM_Q1_0] == (32, 6)
    assert gguf.GGML_QUANT_SIZES[PRISM_Q1_0_G128] == (128, 18)

    q1_block = _encode_prism_block(
        2.0,
        [1, 0] * 16,
        block_size=32,
    ).reshape(1, 6)
    q1 = gguf_quants.dequantize(q1_block, PRISM_Q1_0)
    assert q1.shape == (1, 32)
    assert np.allclose(q1[0, :4], np.array([2.0, -2.0, 2.0, -2.0], dtype=np.float32))

    q1_g128_block = _encode_prism_block(
        0.5,
        [1] * 64 + [0] * 64,
        block_size=128,
    ).reshape(1, 18)
    q1_g128 = gguf_quants.dequantize(q1_g128_block, PRISM_Q1_0_G128)
    assert q1_g128.shape == (1, 128)
    assert np.allclose(q1_g128[0, :4], np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
    assert np.allclose(
        q1_g128[0, -4:], np.array([-0.5, -0.5, -0.5, -0.5], dtype=np.float32)
    )


def test_prism_bonsai_reader_remaps_disk_types_when_model_exists():
    ensure_prism_gguf_compat()

    bonsai = Path(
        "/home/local/Projects/models/registry/PrismML/Bonsai-1.7B-gguf/Bonsai-1.7B.gguf"
    )
    if not bonsai.exists():
        pytest.skip("Local Bonsai GGUF not present")

    reader = gguf.GGUFReader(str(bonsai))
    tensor_types = {int(t.tensor_type) for t in reader.tensors}
    assert PRISM_Q1_0 in tensor_types or PRISM_Q1_0_G128 in tensor_types
