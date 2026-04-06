import logging
import math
import os
import warnings
from enum import IntEnum
from typing import List, Optional

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if _is_cuda or _is_hip:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def _should_use_pytorch_tree_ops() -> bool:
    """Determine whether to use PyTorch fallback for EAGLE tree ops.

    Returns True when sgl_kernel's HIP-ified tree kernels are known to crash
    (e.g. gfx1030/gfx1031/gfx1032) or when the user explicitly requests it
    via SGLANG_USE_PYTORCH_TREE_OPS=1.
    """
    env = os.environ.get("SGLANG_USE_PYTORCH_TREE_OPS", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    # Auto-detect: only needed on HIP (ROCm)
    if not _is_hip:
        return False
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return False
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = getattr(props, "gcnArchName", "")
        if arch.startswith("gfx103"):
            logger.info(
                "Detected %s — using PyTorch fallback for EAGLE tree ops", arch
            )
            return True
    except Exception:
        pass
    return False


_use_pytorch_tree_ops: Optional[bool] = None


def use_pytorch_tree_ops() -> bool:
    """Lazy-cached check for PyTorch tree-op fallback."""
    global _use_pytorch_tree_ops
    if _use_pytorch_tree_ops is None:
        _use_pytorch_tree_ops = _should_use_pytorch_tree_ops()
    return _use_pytorch_tree_ops


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback for build_tree_efficient CUDA kernel
# ---------------------------------------------------------------------------
def _build_tree_pytorch(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
) -> None:
    """Pure-PyTorch translation of the build_tree_efficient CUDA kernel.

    Operates on tensors in-place, matching the exact semantics of the
    ``build_tree_efficient`` and ``build_tree_efficient_partial_packed``
    kernels in ``eagle_utils.cu``.
    """
    bs = parent_list.shape[0]
    # Pull small metadata to CPU for fast scalar access
    pl_cpu = parent_list.cpu()
    si_cpu = selected_index.cpu()
    vsl_cpu = verified_seq_len.cpu()
    si_cols = si_cpu.shape[1]  # == draft_token_num - 1

    is_bitpacking = tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING

    # For bitpacking we need a byte-level view of tree_mask
    if is_bitpacking:
        if draft_token_num > 16:
            num_bytes_per_item = 4
        elif draft_token_num > 8:
            num_bytes_per_item = 2
        else:
            num_bytes_per_item = 1
        tm_bytes = tree_mask.view(torch.uint8)

    # Pre-compute per-batch seq_tree_idx for FULL_MASK
    seq_tree_offsets = []
    if tree_mask_mode == TreeMaskMode.FULL_MASK:
        running = 0
        for b in range(bs):
            seq_tree_offsets.append(
                draft_token_num * draft_token_num * b + running
            )
            running += vsl_cpu[b].item() * draft_token_num

    # Small CPU buffers for retrive_* and positions (avoid per-element GPU sync)
    ri_cpu = retrive_index.cpu()
    rnt_cpu = retrive_next_token.cpu()
    rns_cpu = retrive_next_sibling.cpu()
    pos_cpu = positions.cpu()

    for bid in range(bs):
        seq_len = vsl_cpu[bid].item()

        # ---- tid == 0 : positions[0], retrive_* ----
        pos_cpu[bid * draft_token_num] = seq_len
        ri_offset = bid * draft_token_num

        for i in range(draft_token_num - 1, 0, -1):
            ri_cpu[bid, i] = ri_offset + i
            parent_tb_idx = si_cpu[bid, i - 1].item() // topk
            parent_position = 0

            if parent_tb_idx > 0:
                parent_token_idx = pl_cpu[bid, parent_tb_idx].item()
                found = False
                for pp in range(si_cols):
                    if si_cpu[bid, pp].item() == parent_token_idx:
                        parent_position = pp + 1  # critical +1 shift
                        found = True
                        break
                if not found:
                    parent_position = draft_token_num

            if parent_position == draft_token_num:
                warnings.warn(
                    "WARNING: invalid eagle tree!!! Detected a token with no "
                    "parent token selected. The token will be ignored."
                )
                continue

            existing = rnt_cpu[bid, parent_position].item()
            if existing == -1:
                rnt_cpu[bid, parent_position] = i
            else:
                rnt_cpu[bid, parent_position] = i
                rns_cpu[bid, i] = existing

        ri_cpu[bid, 0] = bid * draft_token_num

        # ---- all tids : tree_mask & positions ----
        for tid in range(draft_token_num):
            if is_bitpacking:
                token_tree_idx = (bid * draft_token_num + tid) * num_bytes_per_item
                # Set bit 0 (root) in first byte
                tm_bytes[token_tree_idx] = 1
            elif tree_mask_mode == TreeMaskMode.FULL_MASK:
                seq_tree_idx = seq_tree_offsets[bid]
                tti = (
                    seq_tree_idx
                    + (seq_len + draft_token_num) * tid
                    + seq_len
                    + 1
                )
                # Root column (already True from fill, but set explicitly)
                tree_mask[tti - 1] = True
                # Clear draft token columns
                tree_mask[tti : tti + draft_token_num - 1] = False
            else:  # QLEN_ONLY
                tti = (
                    draft_token_num * draft_token_num * bid
                    + draft_token_num * tid
                    + 1
                )
                tree_mask[tti - 1] = True
                tree_mask[tti : tti + draft_token_num - 1] = False

            if tid > 0:
                cur_position = tid - 1
                position = 0
                while True:
                    position += 1
                    if is_bitpacking:
                        byte_idx = (cur_position + 1) // 8
                        bit_idx = (cur_position + 1) % 8
                        old = tm_bytes[token_tree_idx + byte_idx].item()
                        tm_bytes[token_tree_idx + byte_idx] = old | (1 << bit_idx)
                    else:
                        tree_mask[tti + cur_position] = True

                    parent_tb_idx = si_cpu[bid, cur_position].item() // topk
                    if parent_tb_idx == 0:
                        break

                    token_idx = pl_cpu[bid, parent_tb_idx].item()
                    found = False
                    for cp in range(si_cols):
                        if si_cpu[bid, cp].item() == token_idx:
                            cur_position = cp
                            found = True
                            break
                    if not found:
                        break

                pos_cpu[bid * draft_token_num + tid] = position + seq_len

    # Copy CPU results back to device
    positions.copy_(pos_cpu.to(positions.device))
    retrive_index.copy_(ri_cpu.to(retrive_index.device))
    retrive_next_token.copy_(rnt_cpu.to(retrive_next_token.device))
    retrive_next_sibling.copy_(rns_cpu.to(retrive_next_sibling.device))


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback for VerifyTreeGreedy CUDA kernel
# ---------------------------------------------------------------------------
def _verify_tree_greedy_pytorch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    """Pure-PyTorch translation of the VerifyTreeGreedy CUDA kernel.

    Modifies *predicts*, *accept_index*, and *accept_token_num* in-place.
    """
    batch_size = candidates.shape[0]
    num_draft_tokens = candidates.shape[1]
    num_speculative_tokens = accept_index.shape[1]

    # Work on CPU for scalar-heavy loop
    ri = retrive_index.cpu()
    rnt = retrive_next_token.cpu()
    rns = retrive_next_sibling.cpu()
    cands = candidates.cpu()
    tp = target_predict.cpu()

    pred_cpu = predicts.cpu()
    ai_cpu = accept_index.cpu()
    atn_cpu = accept_token_num.cpu()

    for bx in range(batch_size):
        last_accepted_ri = ri[bx, 0].item()
        ai_cpu[bx, 0] = last_accepted_ri
        num_accepted = 0
        cur_index = 0

        for j in range(1, num_speculative_tokens):
            cur_index = rnt[bx, cur_index].item()
            while cur_index != -1:
                draft_index = ri[bx, cur_index].item()
                draft_token_id = cands[bx, cur_index].item()
                target_token_id = tp.view(-1)[last_accepted_ri].item()

                if draft_token_id == target_token_id:
                    pred_cpu.view(-1)[last_accepted_ri] = target_token_id
                    num_accepted += 1
                    ai_cpu[bx, num_accepted] = draft_index
                    last_accepted_ri = draft_index
                    break
                else:
                    cur_index = rns[bx, cur_index].item()

            if cur_index == -1:
                break

        atn_cpu[bx] = num_accepted
        pred_cpu.view(-1)[last_accepted_ri] = tp.view(-1)[last_accepted_ri].item()

    predicts.copy_(pred_cpu.to(predicts.device))
    accept_index.copy_(ai_cpu.to(accept_index.device))
    accept_token_num.copy_(atn_cpu.to(accept_token_num.device))


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrive_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    elif use_pytorch_tree_ops():
        _build_tree_pytorch(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
    elif use_pytorch_tree_ops():
        _verify_tree_greedy_pytorch(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
    elif _is_cuda or _is_hip:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num
