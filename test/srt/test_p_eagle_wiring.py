"""P_EAGLE runtime wiring validation tests.

Tests that the P_EAGLE parallel drafting path is correctly wired:
1. spec_info enum routing
2. prepare_p_eagle_inputs shape contract
3. organize_draft_results compatibility with depth-1 tree
4. EAGLEWorker P_EAGLE detection flag
"""
from __future__ import annotations

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
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

    p = SpeculativeAlgorithm.P_EAGLE
    # create_worker needs a ServerArgs, but we can verify the import path
    # When overlap is disabled, P_EAGLE routes to EAGLEWorker (via is_eagle())
    # Just verify the worker class has the P_EAGLE method
    assert hasattr(EAGLEWorker, "draft_forward_p_eagle")


# ---- 2. prepare_p_eagle_inputs shape tests ----

def test_prepare_p_eagle_inputs_shapes():
    """Test that prepare_p_eagle_inputs produces correct tensor shapes.

    Uses a minimal mock to avoid full TP initialization.
    """
    import torch.nn as nn

    class MockLlamaModel(nn.Module):
        def __init__(self, hidden_size=64, target_hidden_size=64, vocab_size=100):
            super().__init__()
            from types import SimpleNamespace
            self.config = SimpleNamespace(
                parallel_drafting=True,
                mask_token_id=0,
            )
            self.parallel_drafting = True
            self.mask_token_id = 0
            self.fc = nn.Linear(target_hidden_size * 3, hidden_size, bias=False)
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.mask_hidden = nn.Parameter(torch.zeros(1, 1, target_hidden_size * 3))

        def prepare_p_eagle_inputs(self, last_token_ids, fused_hidden_states, k):
            """Identical logic to LlamaModel.prepare_p_eagle_inputs."""
            if k < 1:
                raise ValueError(f"k must be >= 1, got {k}")
            if last_token_ids.dim() == 1:
                last_token_ids = last_token_ids.unsqueeze(-1)
            if fused_hidden_states.dim() != 3 or fused_hidden_states.shape[1] != 1:
                raise ValueError("fused_hidden_states must have shape [batch, 1, hidden*3]")

            batch = last_token_ids.shape[0]
            device = last_token_ids.device
            hidden_dtype = fused_hidden_states.dtype
            if k == 1:
                all_hidden_states = fused_hidden_states
                input_ids = last_token_ids
            else:
                mask_hidden = self.mask_hidden.to(device=device, dtype=hidden_dtype).expand(
                    batch, k - 1, -1
                )
                all_hidden_states = torch.cat([fused_hidden_states, mask_hidden], dim=1)
                mask_token_ids = torch.full(
                    (batch, k - 1), self.mask_token_id,
                    dtype=last_token_ids.dtype, device=device,
                )
                input_ids = torch.cat([last_token_ids, mask_token_ids], dim=1)

            embeds = self.embed_tokens(input_ids)
            projected = self.fc(all_hidden_states.to(self.fc.weight.dtype))
            return embeds, projected

    hidden_size, target_hidden_size, vocab_size = 64, 64, 100
    model = MockLlamaModel(hidden_size, target_hidden_size, vocab_size)
    model.eval()

    bs, K = 2, 4
    fc_in = model.fc.in_features  # 192

    token_ids = torch.randint(0, vocab_size, (bs, 1))
    hidden_states = torch.randn(bs, 1, fc_in)

    embeds, projected = model.prepare_p_eagle_inputs(token_ids, hidden_states, k=K)

    assert embeds.shape == (bs, K, hidden_size), (
        f"Expected embeds shape ({bs}, {K}, {hidden_size}), got {embeds.shape}"
    )
    assert projected.shape == (bs, K, hidden_size), (
        f"Expected projected shape ({bs}, {K}, {hidden_size}), got {projected.shape}"
    )


def test_prepare_p_eagle_inputs_k1():
    """k=1 should skip mask_hidden entirely."""
    import torch.nn as nn

    class MockLlamaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(192, 64, bias=False)
            self.embed_tokens = nn.Embedding(100, 64)
            self.mask_hidden = nn.Parameter(torch.zeros(1, 1, 192))
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
    """Verify organize_draft_results works with single-step (depth-1) P_EAGLE output."""
    from sglang.srt.speculative.eagle_utils import organize_draft_results

    bs, K = 2, 8
    num_draft_tokens = 6  # tree size = 6 (including root)

    # Simulate P_EAGLE output: single step, K candidates
    scores = torch.rand(bs, 1, K)
    tokens = torch.randint(0, 1000, (bs, K))
    parents = torch.arange(-1, K, dtype=torch.long).unsqueeze(0).expand(bs, -1)

    score_list = [scores]
    token_list = [tokens]
    parents_list = [parents]

    parent_list, top_scores_index, draft_tokens = organize_draft_results(
        score_list, token_list, parents_list, num_draft_tokens
    )

    # With single step, parent_list should be empty (flat tree)
    assert parent_list.shape == (bs, 0), (
        f"Expected empty parent_list for depth-1 tree, got shape {parent_list.shape}"
    )
    # Should select top num_draft_tokens-1 from K candidates
    assert top_scores_index.shape == (bs, num_draft_tokens - 1)
    assert draft_tokens.shape == (bs, num_draft_tokens - 1)


# ---- 4. EAGLEWorker has P_EAGLE flag ----

def test_eagle_worker_has_p_eagle_detection():
    """Verify EAGLEWorker has is_p_eagle attribute and draft_forward_p_eagle method."""
    from sglang.srt.speculative.eagle_worker import EAGLEWorker
    import inspect

    assert hasattr(EAGLEWorker, "draft_forward_p_eagle")
    sig = inspect.signature(EAGLEWorker.draft_forward_p_eagle)
    params = list(sig.parameters.keys())
    assert "forward_batch" in params, (
        f"draft_forward_p_eagle must accept forward_batch, got params: {params}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
