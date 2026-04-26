"""P_EAGLE runtime wiring validation tests."""

from __future__ import annotations

import inspect

import pytest
import torch


# ---- 1. Enum routing tests ----


def test_p_eagle_enum_routing():
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    p = SpeculativeAlgorithm.P_EAGLE
    assert p.is_eagle(), "P_EAGLE must be recognized as EAGLE family"
    assert p.is_eagle3(), "P_EAGLE must be recognized as EAGLE3 variant"
    assert p.is_p_eagle(), "P_EAGLE must identify as P_EAGLE"
    assert p.needs_draft_model(), "P_EAGLE needs a draft model"

    e3 = SpeculativeAlgorithm.EAGLE3
    assert not e3.is_p_eagle(), "EAGLE3 must NOT identify as P_EAGLE"


def test_p_eagle_creates_eagle_worker():
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

    assert hasattr(EAGLEWorker, "draft_forward_p_eagle")


# ---- 2. prepare_p_eagle_inputs shape tests ----


def test_prepare_p_eagle_inputs_shapes():
    """Test that prepare_p_eagle_inputs produces correct tensor shapes."""
    import torch.nn as nn

    class MockLlamaModel(nn.Module):
        def __init__(self, hidden_size=64, target_hidden_size=64, vocab_size=100):
            super().__init__()
            from types import SimpleNamespace

            self.config = SimpleNamespace(parallel_drafting=True, mask_token_id=0)
            self.parallel_drafting = True
            self.mask_token_id = 0
            self.fc = nn.Linear(target_hidden_size * 3, hidden_size, bias=False)
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.mask_hidden = nn.Parameter(torch.zeros(1, 1, target_hidden_size * 3))

        def prepare_p_eagle_inputs(self, last_token_ids, fused_hidden_states, k):
            if k < 1:
                raise ValueError(f"k must be >= 1, got {k}")
            if last_token_ids.dim() == 1:
                last_token_ids = last_token_ids.unsqueeze(-1)
            if fused_hidden_states.dim() != 3 or fused_hidden_states.shape[1] != 1:
                raise ValueError(
                    "fused_hidden_states must have shape [batch, 1, hidden*3]"
                )

            batch = last_token_ids.shape[0]
            device = last_token_ids.device
            hidden_dtype = fused_hidden_states.dtype
            if k == 1:
                all_hidden_states = fused_hidden_states
                input_ids = last_token_ids
            else:
                mask_hidden = self.mask_hidden.to(
                    device=device, dtype=hidden_dtype
                ).expand(batch, k - 1, -1)
                all_hidden_states = torch.cat([fused_hidden_states, mask_hidden], dim=1)
                mask_token_ids = torch.full(
                    (batch, k - 1),
                    self.mask_token_id,
                    dtype=last_token_ids.dtype,
                    device=device,
                )
                input_ids = torch.cat([last_token_ids, mask_token_ids], dim=1)

            embeds = self.embed_tokens(input_ids)
            projected = self.fc(all_hidden_states.to(self.fc.weight.dtype))
            return embeds, projected

    hidden_size, target_hidden_size, vocab_size = 64, 64, 100
    model = MockLlamaModel(hidden_size, target_hidden_size, vocab_size)
    model.eval()

    bs, k = 2, 4
    fc_in = model.fc.in_features
    token_ids = torch.randint(0, vocab_size, (bs, 1))
    hidden_states = torch.randn(bs, 1, fc_in)

    embeds, projected = model.prepare_p_eagle_inputs(token_ids, hidden_states, k=k)

    assert embeds.shape == (bs, k, hidden_size)
    assert projected.shape == (bs, k, hidden_size)


def test_prepare_p_eagle_inputs_k1():
    """k=1 should skip mask_hidden entirely."""
    import torch.nn as nn

    class MockLlamaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(192, 64, bias=False)
            self.embed_tokens = nn.Embedding(100, 64)
            self.mask_token_id = 0

        def prepare_p_eagle_inputs(self, last_token_ids, fused_hidden_states, k):
            if last_token_ids.dim() == 1:
                last_token_ids = last_token_ids.unsqueeze(-1)
            if k == 1:
                embeds = self.embed_tokens(last_token_ids)
                projected = self.fc(fused_hidden_states.to(self.fc.weight.dtype))
                return embeds, projected
            raise NotImplementedError("k>1 not tested here")

    model = MockLlamaModel()
    model.eval()

    bs = 3
    token_ids = torch.randint(0, 100, (bs, 1))
    hidden_states = torch.randn(bs, 1, 192)

    embeds, projected = model.prepare_p_eagle_inputs(token_ids, hidden_states, k=1)

    assert embeds.shape == (bs, 1, 64)
    assert projected.shape == (bs, 1, 64)


# ---- 3. organize_draft_results depth-1 tree compatibility ----


def test_organize_draft_results_depth1_tree():
    """Verify organize_draft_results works with single-step P_EAGLE output."""
    from sglang.srt.speculative.eagle_utils import organize_draft_results

    bs, k = 2, 8
    num_draft_tokens = 6

    scores = torch.rand(bs, 1, k)
    tokens = torch.randint(0, 1000, (bs, k))
    parents = torch.arange(-1, k, dtype=torch.long).unsqueeze(0).expand(bs, -1)

    parent_list, top_scores_index, draft_tokens = organize_draft_results(
        [scores], [tokens], [parents], num_draft_tokens
    )

    assert parent_list.shape == (bs, 0)
    assert top_scores_index.shape == (bs, num_draft_tokens - 1)
    assert draft_tokens.shape == (bs, num_draft_tokens - 1)


# ---- 4. EAGLEWorker has P_EAGLE flag ----


def test_eagle_worker_has_p_eagle_detection():
    """Verify EAGLEWorker has P_EAGLE detection method."""
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

    assert hasattr(EAGLEWorker, "draft_forward_p_eagle")
    sig = inspect.signature(EAGLEWorker.draft_forward_p_eagle)
    assert "forward_batch" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
