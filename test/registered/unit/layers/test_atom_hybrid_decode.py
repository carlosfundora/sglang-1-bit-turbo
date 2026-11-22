import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.hybrid_atom_backend import HybridAITERTritonBackend


class _RecordingBackend:
    def __init__(self):
        self.last_call = None

    def forward(self, query, key, value, **kwargs):
        self.last_call = (query, key, value, kwargs)
        return torch.empty_like(query)


class TestAtomHybridDecode(unittest.TestCase):
    def test_decode_routes_to_aiter_and_preserves_kv_references(self):
        with patch.object(HybridAITERTritonBackend, "_init_backends", lambda self: None):
            backend = HybridAITERTritonBackend(runner=object())

        recorder = _RecordingBackend()
        backend.aiter_backend = recorder
        backend.triton_backend = _RecordingBackend()

        query = torch.randn(1, 1, 8, 64)
        key = torch.randn(1, 16, 8, 64)
        value = torch.randn(1, 16, 8, 64)

        backend.forward(query, key, value, is_prefill=False)

        assert recorder.last_call is not None
        _, passed_key, passed_value, kwargs = recorder.last_call
        assert passed_key is key
        assert passed_value is value
        assert kwargs.get("kv_seq_len") is None

