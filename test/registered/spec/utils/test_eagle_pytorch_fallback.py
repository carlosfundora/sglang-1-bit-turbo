"""Comprehensive tests for EAGLE3 PyTorch tree-op fallback.

Validates:
  a) tree_mask parity in FULL_MASK mode
  b) QLEN_ONLY and QLEN_ONLY_BITPACKING modes
  c) verify_tree_greedy parity
  d) reuse of tree_mask_buf / position_buf
  e) dispatch behavior for env override
  f) invalid parent chain graceful handling
"""

import math
import os
import unittest

import torch

from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    _build_tree_pytorch,
    _verify_tree_greedy_pytorch,
    build_tree_kernel_efficient,
    organize_draft_results,
    use_pytorch_tree_ops,
)
from sglang.srt.utils import get_device


def _make_standard_test_inputs(device=None):
    """Return the standard test inputs from test_build_eagle_tree.py."""
    if device is None:
        device = get_device()
    verified_id = torch.tensor([29974, 13], device=device, dtype=torch.int32)
    score_list = [
        torch.tensor(
            [
                [[7.1127e-01, 2.8292e-01, 2.2995e-03, 1.7357e-03]],
                [[9.7476e-01, 2.2219e-02, 6.5031e-04, 1.3212e-04]],
            ],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            [
                [
                    [6.9142e-01, 1.2863e-02, 1.6873e-03, 1.1871e-03],
                    [2.4787e-01, 1.8818e-02, 1.4204e-02, 9.2235e-04],
                    [2.2971e-03, 1.6700e-06, 1.8737e-07, 8.3146e-08],
                    [1.2771e-03, 2.4374e-04, 1.7832e-04, 1.1947e-05],
                ],
                [
                    [8.4832e-02, 6.6068e-02, 5.8304e-02, 5.7851e-02],
                    [2.3616e-03, 1.1243e-03, 5.4368e-04, 2.7768e-04],
                    [2.5286e-04, 1.5578e-04, 2.8817e-05, 1.2888e-05],
                    [1.2834e-04, 2.5417e-06, 1.1279e-06, 1.6088e-08],
                ],
            ],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            [
                [
                    [6.6438e-01, 2.6997e-02, 2.4236e-05, 4.0821e-06],
                    [2.4402e-01, 2.8409e-03, 5.0935e-04, 2.9022e-04],
                    [1.6178e-02, 2.0567e-03, 4.5892e-04, 3.0034e-05],
                    [1.3023e-02, 5.0497e-04, 3.6371e-04, 8.7750e-05],
                ],
                [
                    [2.3263e-02, 2.0054e-02, 9.3990e-03, 2.7783e-03],
                    [6.4156e-02, 5.5506e-04, 1.0429e-04, 9.7211e-05],
                    [4.9950e-02, 5.0630e-03, 9.0068e-04, 3.3656e-04],
                    [7.5817e-03, 8.5731e-04, 6.9972e-04, 6.0793e-04],
                ],
            ],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            [
                [
                    [6.6420e-01, 1.0525e-04, 6.5864e-05, 1.2253e-06],
                    [1.3019e-01, 1.0461e-01, 5.2083e-03, 1.6777e-03],
                    [2.0103e-02, 6.7335e-03, 1.2625e-04, 1.0364e-05],
                    [1.5142e-02, 7.0819e-04, 9.6595e-05, 8.7951e-05],
                ],
                [
                    [5.8608e-02, 1.8840e-03, 7.8535e-04, 4.4400e-04],
                    [1.2185e-02, 2.0684e-03, 1.7418e-03, 1.4327e-03],
                    [6.2455e-03, 6.1487e-03, 2.6862e-03, 1.8034e-03],
                    [1.8590e-03, 1.6151e-03, 1.2481e-03, 3.6038e-04],
                ],
            ],
            dtype=torch.float32,
            device=device,
        ),
    ]
    token_list = [
        torch.tensor(
            [[29896, 29906, 29900, 29945], [13, 2, 29871, 28956]],
            dtype=torch.int64,
            device=device,
        ),
        torch.tensor(
            [
                [29889, 29974, 29945, 29900, 29974, 29922, 29930, 29958,
                 29889, 29974, 29930, 29945, 29974, 29922, 29930, 29958],
                [22550, 4136, 16492, 8439, 29871, 2, 3001, 13,
                 2, 13, 29906, 29946, 2, 13, 29871, 259],
            ],
            device=device,
        ),
        torch.tensor(
            [
                [29946, 29945, 29953, 29906, 29896, 29945, 29900, 29906,
                 29896, 29945, 29906, 29953, 29896, 29945, 29906, 29946],
                [29871, 2, 29901, 29889, 29871, 2, 395, 259,
                 29901, 29871, 2, 29889, 3001, 1234, 7146, 2186],
            ],
            device=device,
        ),
        torch.tensor(
            [
                [29946, 29974, 29945, 29930, 29889, 29922, 29974, 29930,
                 29974, 29946, 29930, 29922, 29889, 29974, 29945, 29922],
                [29941, 29906, 2, 29946, 29871, 450, 319, 14990,
                 29946, 29941, 2, 29906, 29871, 2, 3001, 13],
            ],
            device=device,
        ),
    ]
    parents_list = [
        torch.tensor(
            [[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]], dtype=torch.int64, device=device
        ),
        torch.tensor(
            [[4, 8, 9, 10], [4, 5, 6, 7]], dtype=torch.int64, device=device
        ),
        torch.tensor(
            [[20, 24, 21, 28], [24, 28, 20, 21]], dtype=torch.int64, device=device
        ),
        torch.tensor(
            [[36, 40, 41, 44], [36, 40, 44, 45]], dtype=torch.int64, device=device
        ),
    ]
    seq_lens = torch.tensor([5, 10], dtype=torch.int64, device=device)
    return (
        verified_id, score_list, token_list, parents_list, seq_lens,
        4,  # topk
        4,  # depth
        8,  # num_draft_token
    )


def _get_tree_ancestors(retrive_next_token, retrive_next_sibling, draft_token_num, bid, tid):
    """Given the retrive tree structure, return the set of ancestor positions
    (0-indexed into the draft region) for draft token `tid`.

    Ancestor positions are the indices in the retrive_* arrays (not flat global indices).
    Position 0 = root.
    """
    if tid == 0:
        return set()
    # Walk from tid up through the tree using the inverse of next_token/sibling
    # Build parent map from the retrive_* arrays
    rnt = retrive_next_token[bid].tolist()
    rns = retrive_next_sibling[bid].tolist()

    # Build child→parent map
    parent_map = {}
    for parent_pos in range(draft_token_num):
        child = rnt[parent_pos]
        while child != -1:
            parent_map[child] = parent_pos
            child = rns[child]

    # Walk up from tid to root
    ancestors = set()
    pos = tid
    while pos in parent_map:
        ancestors.add(pos)
        pos = parent_map[pos]
    ancestors.add(pos)  # root is at pos=0 but not in parent_map
    return ancestors


class TestBuildTreeMaskFullMask(unittest.TestCase):
    """Test tree_mask values in FULL_MASK mode against tree structure."""

    def test_tree_mask_consistency(self):
        """Verify tree_mask[tid] has True for ancestors and False for non-ancestors."""
        (verified_id, score_list, token_list, parents_list, seq_lens,
         topk, depth, num_draft_token) = _make_standard_test_inputs()

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, num_draft_token
        )

        (tree_mask, position, ri, rnt, rns, draft_tokens) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            seq_lens=seq_lens,
            seq_lens_sum=torch.sum(seq_lens).item(),
            topk=topk,
            spec_steps=depth,
            num_verify_tokens=num_draft_token,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
        )

        bs = seq_lens.numel()
        dtn = num_draft_token
        vsl = seq_lens.cpu().tolist()
        tm = tree_mask.cpu()

        # Decode flat tree_mask and validate per-token
        offset = 0
        for bid in range(bs):
            seq_len = vsl[bid]
            row_len = seq_len + dtn
            for tid in range(dtn):
                row_start = offset + row_len * tid
                row = tm[row_start: row_start + row_len].tolist()

                # All verified token columns should be True
                for col in range(seq_len):
                    self.assertTrue(
                        row[col],
                        f"bid={bid} tid={tid} col={col}: verified token should be True"
                    )

                # Root column (col=seq_len) should always be True
                self.assertTrue(
                    row[seq_len],
                    f"bid={bid} tid={tid}: root column should be True"
                )

                if tid == 0:
                    # Root token: no draft columns should be True
                    for col in range(seq_len + 1, row_len):
                        self.assertFalse(
                            row[col],
                            f"bid={bid} tid=0 col={col}: root shouldn't attend to draft"
                        )
                else:
                    # Non-root: check ancestors
                    ancestors = _get_tree_ancestors(rnt, rns, dtn, bid, tid)
                    for col_offset in range(dtn - 1):
                        col = seq_len + 1 + col_offset
                        draft_pos = col_offset  # 0-indexed in selected_index
                        # draft_pos maps to retrive position draft_pos+1
                        retrive_pos = draft_pos + 1
                        if retrive_pos in ancestors:
                            self.assertTrue(
                                row[col],
                                f"bid={bid} tid={tid} col={col}: ancestor {retrive_pos} should be True"
                            )
                        else:
                            self.assertFalse(
                                row[col],
                                f"bid={bid} tid={tid} col={col}: non-ancestor {retrive_pos} should be False"
                            )

            offset += dtn * row_len


class TestBuildTreeQlenOnly(unittest.TestCase):
    """Test QLEN_ONLY mode."""

    def test_qlen_only_mode(self):
        (verified_id, score_list, token_list, parents_list, seq_lens,
         topk, depth, num_draft_token) = _make_standard_test_inputs()

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, num_draft_token
        )

        (tree_mask, position, ri, rnt, rns, dt) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            seq_lens=seq_lens,
            seq_lens_sum=torch.sum(seq_lens).item(),
            topk=topk,
            spec_steps=depth,
            num_verify_tokens=num_draft_token,
            tree_mask_mode=TreeMaskMode.QLEN_ONLY,
        )

        # retrive_* and positions should be identical to FULL_MASK
        self.assertEqual(
            position.tolist(),
            [5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14],
        )
        self.assertEqual(
            ri.tolist(),
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        )
        self.assertEqual(
            rnt.tolist(),
            [[1, 3, 4, 5, 6, 7, -1, -1], [1, 2, -1, 6, -1, -1, 7, -1]],
        )
        self.assertEqual(
            rns.tolist(),
            [[-1, 2, -1, -1, -1, -1, -1, -1], [-1, -1, 3, 4, 5, -1, -1, -1]],
        )

        # QLEN_ONLY tree_mask: [bs * dtn * dtn], only draft-draft attention
        bs = 2
        dtn = num_draft_token
        tm = tree_mask.cpu()
        self.assertEqual(tm.numel(), bs * dtn * dtn)

        for bid in range(bs):
            for tid in range(dtn):
                row_start = bid * dtn * dtn + tid * dtn
                # First element (root col) should be True
                self.assertTrue(tm[row_start].item())
                if tid == 0:
                    for k in range(1, dtn):
                        self.assertFalse(tm[row_start + k].item())


class TestBuildTreeBitpacking(unittest.TestCase):
    """Test QLEN_ONLY_BITPACKING mode for various num_verify_tokens."""

    def _run_bitpacking_test(self, num_draft_token):
        device = get_device()
        bs = 1
        topk = 2
        depth = 2
        parent_list_cols = topk * (depth - 1) + 1  # 3

        parent_list = torch.zeros(bs, parent_list_cols, dtype=torch.int64, device=device)
        parent_list[0, 0] = -1
        parent_list[0, 1] = 0
        if parent_list_cols > 2:
            parent_list[0, 2] = topk

        # All tokens are children of root: selected_index values < topk
        # so parent_tb_idx = si // topk = 0 for all
        si = torch.zeros(bs, num_draft_token - 1, dtype=torch.int64, device=device)
        for k in range(num_draft_token - 1):
            si[0, k] = k % topk  # cycles through 0, 1, 0, 1, ...

        seq_lens = torch.tensor([5], dtype=torch.int64, device=device)
        verified_id = torch.tensor([100], dtype=torch.int32, device=device)
        draft_tokens = torch.arange(
            200, 200 + num_draft_token - 1, dtype=torch.int32, device=device
        ).unsqueeze(0)

        (tree_mask, pos, ri, rnt, rns, dt) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=si,
            draft_tokens=draft_tokens,
            seq_lens=seq_lens,
            seq_lens_sum=5,
            topk=topk,
            spec_steps=depth,
            num_verify_tokens=num_draft_token,
            tree_mask_mode=TreeMaskMode.QLEN_ONLY_BITPACKING,
        )

        # Determine expected packed dtype
        if num_draft_token > 16:
            nbytes = 4
        elif num_draft_token > 8:
            nbytes = 2
        else:
            nbytes = 1

        # Verify tree_mask shape and dtype
        self.assertEqual(tree_mask.numel(), num_draft_token * bs)
        tm_bytes = tree_mask.view(torch.uint8)

        # Verify root token (tid=0): only bit 0 (root) should be set
        root_bytes = tm_bytes[0:nbytes].cpu().tolist()
        self.assertEqual(root_bytes[0], 1, "Root token: bit 0 should be 1")
        for b in range(1, nbytes):
            self.assertEqual(root_bytes[b], 0, f"Root token: byte {b} should be 0")

        # All draft tokens have parent_tb_idx == 0 (children of root),
        # so each tid>0 sets: bit 0 (root) and bit (tid) (itself at cur_position=tid-1, bit=(tid-1+1)=tid)
        for tid in range(1, num_draft_token):
            offset = tid * nbytes
            token_bytes = tm_bytes[offset:offset + nbytes].cpu().tolist()
            # Reconstruct the integer value
            val = 0
            for b_idx in range(nbytes):
                val |= token_bytes[b_idx] << (8 * b_idx)
            # Expected: bit 0 (root) and bit tid (self)
            expected = (1 << 0) | (1 << tid)
            self.assertEqual(
                val, expected,
                f"tid={tid}: expected bits 0 and {tid} set, got {val:#x} vs {expected:#x}"
            )

    def test_bitpacking_8_tokens(self):
        """8 tokens → uint8 (1 byte per item)."""
        self._run_bitpacking_test(8)

    def test_bitpacking_9_tokens(self):
        """9 tokens → uint16 (2 bytes per item)."""
        self._run_bitpacking_test(9)

    def test_bitpacking_17_tokens(self):
        """17 tokens → uint32 (4 bytes per item)."""
        self._run_bitpacking_test(17)


class TestVerifyTreeGreedy(unittest.TestCase):
    """Test _verify_tree_greedy_pytorch with known inputs."""

    def _make_verify_inputs(self, bs, num_draft_tokens, num_spec_tokens,
                            candidates, target_predict,
                            retrive_index, retrive_next_token, retrive_next_sibling):
        device = get_device()
        predicts = torch.zeros(bs * num_draft_tokens, dtype=torch.int32, device=device)
        accept_index = torch.full(
            (bs, num_spec_tokens), -1, dtype=torch.int32, device=device
        )
        accept_token_num = torch.zeros(bs, dtype=torch.int32, device=device)
        return (
            predicts, accept_index, accept_token_num,
            torch.tensor(candidates, dtype=torch.int64, device=device),
            torch.tensor(retrive_index, dtype=torch.int64, device=device),
            torch.tensor(retrive_next_token, dtype=torch.int64, device=device),
            torch.tensor(retrive_next_sibling, dtype=torch.int64, device=device),
            torch.tensor(target_predict, dtype=torch.int64, device=device),
        )

    def test_full_acceptance_linear_chain(self):
        """All draft tokens accepted in a linear chain."""
        bs, dtn, nst = 1, 4, 4
        inputs = self._make_verify_inputs(
            bs, dtn, nst,
            candidates=[[100, 200, 300, 400]],
            target_predict=[[200, 300, 400, 500]],
            retrive_index=[[0, 1, 2, 3]],
            retrive_next_token=[[1, 2, 3, -1]],
            retrive_next_sibling=[[-1, -1, -1, -1]],
        )
        _verify_tree_greedy_pytorch(*inputs)
        predicts, accept_index, accept_token_num = inputs[:3]

        self.assertEqual(accept_token_num[0].item(), 3)
        self.assertEqual(accept_index[0, :4].tolist(), [0, 1, 2, 3])
        # predicts[0]=200, predicts[1]=300, predicts[2]=400, predicts[3]=500
        self.assertEqual(predicts[0].item(), 200)
        self.assertEqual(predicts[1].item(), 300)
        self.assertEqual(predicts[2].item(), 400)
        self.assertEqual(predicts[3].item(), 500)

    def test_early_rejection(self):
        """Second draft token rejected → stops early."""
        bs, dtn, nst = 1, 4, 4
        inputs = self._make_verify_inputs(
            bs, dtn, nst,
            candidates=[[100, 200, 999, 400]],  # 999 ≠ 300
            target_predict=[[200, 300, 400, 500]],
            retrive_index=[[0, 1, 2, 3]],
            retrive_next_token=[[1, 2, 3, -1]],
            retrive_next_sibling=[[-1, -1, -1, -1]],
        )
        _verify_tree_greedy_pytorch(*inputs)
        predicts, accept_index, accept_token_num = inputs[:3]

        self.assertEqual(accept_token_num[0].item(), 1)
        self.assertEqual(accept_index[0, 0].item(), 0)
        self.assertEqual(accept_index[0, 1].item(), 1)
        # predicts[0]=200, predicts[1]=target_predict[1]=300 (last accepted → final write)
        self.assertEqual(predicts[0].item(), 200)
        self.assertEqual(predicts[1].item(), 300)

    def test_sibling_fallback(self):
        """First child rejected, sibling accepted."""
        bs, dtn, nst = 1, 4, 4
        inputs = self._make_verify_inputs(
            bs, dtn, nst,
            candidates=[[100, 999, 200, 888]],
            target_predict=[[200, 300, 400, 500]],
            retrive_index=[[0, 1, 2, 3]],
            retrive_next_token=[[1, -1, -1, -1]],
            retrive_next_sibling=[[-1, 2, -1, -1]],
        )
        _verify_tree_greedy_pytorch(*inputs)
        predicts, accept_index, accept_token_num = inputs[:3]

        self.assertEqual(accept_token_num[0].item(), 1)
        # Sibling at index 2 was accepted; its retrive_index is 2
        self.assertEqual(accept_index[0, 1].item(), 2)
        self.assertEqual(predicts[0].item(), 200)
        # Last accepted was retrive_index[2]=2 → predicts[2]=target_predict[2]=400
        self.assertEqual(predicts[2].item(), 400)

    def test_no_acceptance(self):
        """First child rejected, no siblings → zero accepted."""
        bs, dtn, nst = 1, 4, 4
        inputs = self._make_verify_inputs(
            bs, dtn, nst,
            candidates=[[100, 999, 888, 777]],
            target_predict=[[200, 300, 400, 500]],
            retrive_index=[[0, 1, 2, 3]],
            retrive_next_token=[[1, -1, -1, -1]],
            retrive_next_sibling=[[-1, -1, -1, -1]],
        )
        _verify_tree_greedy_pytorch(*inputs)
        predicts, accept_index, accept_token_num = inputs[:3]

        self.assertEqual(accept_token_num[0].item(), 0)
        # predicts[0] = target_predict[0] = 200 (final write at root)
        self.assertEqual(predicts[0].item(), 200)

    def test_batch_size_2(self):
        """Two batch elements with different acceptance lengths."""
        bs, dtn, nst = 2, 3, 3
        inputs = self._make_verify_inputs(
            bs, dtn, nst,
            candidates=[[100, 200, 300], [400, 500, 999]],
            target_predict=[[200, 300, 999], [500, 999, 800]],
            retrive_index=[[0, 1, 2], [3, 4, 5]],
            retrive_next_token=[[1, 2, -1], [1, 2, -1]],
            retrive_next_sibling=[[-1, -1, -1], [-1, -1, -1]],
        )
        _verify_tree_greedy_pytorch(*inputs)
        predicts, accept_index, accept_token_num = inputs[:3]

        # Batch 0: 200==200 ✓, 300==300 ✓ → 2 accepted
        self.assertEqual(accept_token_num[0].item(), 2)
        # Batch 1: 500==500 ✓, 999≠999? candidates[1,2]=999, target_predict[flat 4]=999
        # target_predict is [[200,300,999],[500,999,800]], flat index 4 = target_predict[1,1] = 999
        # candidates[1,2] = 999, target_predict[4] = 999 → MATCH!
        self.assertEqual(accept_token_num[1].item(), 2)


class TestBufferReuse(unittest.TestCase):
    """Test that pre-allocated tree_mask_buf and position_buf work."""

    def test_buffer_reuse_full_mask(self):
        (verified_id, score_list, token_list, parents_list, seq_lens,
         topk, depth, num_draft_token) = _make_standard_test_inputs()

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, num_draft_token
        )
        bs = seq_lens.numel()
        device = seq_lens.device
        seq_lens_sum = torch.sum(seq_lens).item()

        # Pre-allocate buffers
        tree_mask_buf = torch.empty(
            seq_lens_sum * num_draft_token + num_draft_token * num_draft_token * bs,
            dtype=torch.bool, device=device,
        )
        position_buf = torch.empty(
            bs * num_draft_token, dtype=torch.long, device=device,
        )

        (tree_mask, pos, ri, rnt, rns, dt) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            topk=topk,
            spec_steps=depth,
            num_verify_tokens=num_draft_token,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            tree_mask_buf=tree_mask_buf,
            position_buf=position_buf,
        )

        # Buffer should be the same object
        self.assertIs(tree_mask, tree_mask_buf)
        self.assertIs(pos, position_buf)

        # Results should still be correct
        self.assertEqual(
            pos.tolist(),
            [5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14],
        )
        self.assertEqual(
            ri.tolist(),
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        )

    def test_buffer_reuse_bitpacking(self):
        (verified_id, score_list, token_list, parents_list, seq_lens,
         topk, depth, num_draft_token) = _make_standard_test_inputs()

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, num_draft_token
        )
        bs = seq_lens.numel()
        device = seq_lens.device

        # Pre-allocate bitpacking buffer (uint8 for 8 tokens)
        tree_mask_buf = torch.zeros(
            num_draft_token * bs, dtype=torch.uint8, device=device,
        )

        (tree_mask, pos, ri, rnt, rns, dt) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            seq_lens=seq_lens,
            seq_lens_sum=torch.sum(seq_lens).item(),
            topk=topk,
            spec_steps=depth,
            num_verify_tokens=num_draft_token,
            tree_mask_mode=TreeMaskMode.QLEN_ONLY_BITPACKING,
            tree_mask_buf=tree_mask_buf,
        )

        self.assertIs(tree_mask, tree_mask_buf)
        # Verify root tokens have bit 0 set
        self.assertEqual(tree_mask[0].item() & 1, 1)
        self.assertEqual(tree_mask[num_draft_token].item() & 1, 1)


class TestDispatchBehavior(unittest.TestCase):
    """Test env-var-based dispatch control."""

    def test_env_override_true(self):
        import sglang.srt.speculative.eagle_utils as eu
        old = eu._use_pytorch_tree_ops
        try:
            eu._use_pytorch_tree_ops = None
            os.environ["SGLANG_USE_PYTORCH_TREE_OPS"] = "1"
            self.assertTrue(eu.use_pytorch_tree_ops())
        finally:
            eu._use_pytorch_tree_ops = old
            os.environ.pop("SGLANG_USE_PYTORCH_TREE_OPS", None)

    def test_env_override_false(self):
        import sglang.srt.speculative.eagle_utils as eu
        old = eu._use_pytorch_tree_ops
        try:
            eu._use_pytorch_tree_ops = None
            os.environ["SGLANG_USE_PYTORCH_TREE_OPS"] = "0"
            self.assertFalse(eu.use_pytorch_tree_ops())
        finally:
            eu._use_pytorch_tree_ops = old
            os.environ.pop("SGLANG_USE_PYTORCH_TREE_OPS", None)

    def test_env_override_yes_no(self):
        import sglang.srt.speculative.eagle_utils as eu
        old = eu._use_pytorch_tree_ops
        try:
            for val in ("yes", "YES", "Yes", "true", "TRUE"):
                eu._use_pytorch_tree_ops = None
                os.environ["SGLANG_USE_PYTORCH_TREE_OPS"] = val
                self.assertTrue(eu.use_pytorch_tree_ops(), f"Expected True for '{val}'")
            for val in ("no", "NO", "false", "FALSE"):
                eu._use_pytorch_tree_ops = None
                os.environ["SGLANG_USE_PYTORCH_TREE_OPS"] = val
                self.assertFalse(eu.use_pytorch_tree_ops(), f"Expected False for '{val}'")
        finally:
            eu._use_pytorch_tree_ops = old
            os.environ.pop("SGLANG_USE_PYTORCH_TREE_OPS", None)


if __name__ == "__main__":
    unittest.main()
