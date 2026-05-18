import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.qwen2_audio import (
    _concat_audio_feature_lens_with_preallocation,
    _concat_audio_features_with_preallocation,
    _flatten_projected_audio_embeds_with_preallocation,
)


class TestQwen2AudioPrealloc(unittest.TestCase):
    def test_concat_audio_features_matches_torch_cat(self):
        items = [
            SimpleNamespace(feature=torch.randn(2, 4)),
            SimpleNamespace(feature=torch.randn(3, 4)),
            SimpleNamespace(feature=torch.randn(1, 4)),
        ]
        expected = torch.cat([item.feature for item in items], dim=0).to(torch.float16)
        actual = _concat_audio_features_with_preallocation(items, torch.float16)
        self.assertTrue(torch.equal(actual, expected))

    def test_concat_audio_feature_lens_matches_torch_cat(self):
        items = [
            SimpleNamespace(audio_feature_lens=torch.tensor([2, 1], dtype=torch.int64)),
            SimpleNamespace(audio_feature_lens=torch.tensor([3], dtype=torch.int64)),
            SimpleNamespace(audio_feature_lens=torch.tensor([0, 4], dtype=torch.int64)),
        ]
        expected = torch.cat([item.audio_feature_lens for item in items], dim=0)
        actual = _concat_audio_feature_lens_with_preallocation(items)
        self.assertTrue(torch.equal(actual, expected))

    def test_flatten_projected_audio_embeds_matches_reference(self):
        audio_embeds = torch.randn(3, 5, 8)
        audio_feature_lens = torch.tensor([2, 4, 1], dtype=torch.int64)

        expected = torch.cat(
            [embed[:length.item()] for length, embed in zip(audio_feature_lens, audio_embeds)],
            dim=0,
        )
        actual = _flatten_projected_audio_embeds_with_preallocation(
            audio_embeds, audio_feature_lens
        )
        self.assertTrue(torch.equal(actual, expected))

