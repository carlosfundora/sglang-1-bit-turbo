# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Triton JIT kernel for tree_speculative_sampling_target_only.

Tier-2 fallback: device-agnostic GPU kernel that works on both CUDA and ROCm.
Semantically equivalent to the CUDA/HIP C++ kernel.

One program per batch element.
  Phase 1 — sequential tree walk (tree is tiny, ~4 nodes).
  Phase 2 — parallelised correction sampling over vocabulary in blocks.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tree_spec_sampling_kernel(
    # Mutable outputs
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    # Read-only inputs
    candidates_ptr,
    retrive_index_ptr,
    retrive_next_token_ptr,
    retrive_next_sibling_ptr,
    uniform_samples_ptr,
    uniform_final_ptr,
    target_probs_ptr,
    draft_probs_ptr,
    # Scalar params
    threshold_single: tl.constexpr,
    threshold_acc: tl.constexpr,
    # Shape params
    num_draft_tokens: tl.constexpr,
    num_spec_steps: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    bx = tl.program_id(0)

    # Strides for 2-D [bs, num_draft_tokens] tensors
    row_off = bx * num_draft_tokens

    # Strides for 3-D [bs, num_draft_tokens, vocab_size] tensors
    prob_batch_off = bx * num_draft_tokens * vocab_size

    # ------------------------------------------------------------------
    # Phase 1: Tree walk with probabilistic acceptance
    # ------------------------------------------------------------------
    prob_acc: tl.float32 = 0.0

    # Load initial int64 values to establish types
    last_accepted_retrive_idx = tl.load(retrive_index_ptr + row_off + 0)
    # cur_prob_offset must be int64 to match reassignment from cur_index
    cur_prob_offset = last_accepted_retrive_idx * 0  # int64 zero

    coin = tl.load(uniform_samples_ptr + row_off + 0).to(tl.float32)

    # accept_index[bx, 0] = last_accepted_retrive_idx
    tl.store(accept_index_ptr + bx * num_spec_steps + 0, last_accepted_retrive_idx)
    num_accepted: tl.int32 = 0
    # cur_index must be int64 (from retrive_next_token loads)
    cur_index = last_accepted_retrive_idx * 0  # int64 zero

    threshold_acc_safe: tl.float32 = threshold_acc
    if threshold_acc_safe < 1e-9:
        threshold_acc_safe = 1e-9

    done: tl.int32 = 0  # set to 1 when tree walk terminates early

    for _j in range(1, num_spec_steps):
        if done == 0:
            # Move to next depth
            cur_index = tl.load(retrive_next_token_ptr + row_off + cur_index)

            accepted_this_depth: tl.int32 = 0  # flag
            exhausted_siblings: tl.int32 = 0

            # Walk siblings at this depth (max = num_draft_tokens iterations)
            for _s in range(num_draft_tokens):
                if accepted_this_depth == 0 and exhausted_siblings == 0:
                    if cur_index == -1:
                        exhausted_siblings = 1
                    else:
                        draft_index = tl.load(retrive_index_ptr + row_off + cur_index)
                        draft_token_id = tl.load(candidates_ptr + row_off + cur_index)

                        # target_probs[bx, cur_prob_offset, draft_token_id]
                        tp_addr = (
                            prob_batch_off
                            + cur_prob_offset * vocab_size
                            + draft_token_id
                        )
                        target_prob_single = tl.load(target_probs_ptr + tp_addr).to(
                            tl.float32
                        )
                        prob_acc += target_prob_single

                        if (
                            coin <= prob_acc / threshold_acc_safe
                            or target_prob_single >= threshold_single
                        ):
                            # ACCEPT
                            prob_acc = 0.0
                            cur_prob_offset = cur_index
                            coin = tl.load(
                                uniform_samples_ptr + row_off + cur_index
                            ).to(tl.float32)
                            tl.store(
                                predicts_ptr + last_accepted_retrive_idx,
                                draft_token_id.to(tl.int32),
                            )
                            num_accepted += 1
                            tl.store(
                                accept_index_ptr + bx * num_spec_steps + num_accepted,
                                draft_index.to(tl.int32),
                            )
                            last_accepted_retrive_idx = draft_index
                            accepted_this_depth = 1
                        else:
                            # REJECT — copy target mass into draft_probs for correction
                            tl.store(draft_probs_ptr + tp_addr, target_prob_single)
                            cur_index = tl.load(
                                retrive_next_sibling_ptr + row_off + cur_index
                            )

            # If no sibling was accepted and we exhausted siblings, stop
            if accepted_this_depth == 0:
                done = 1

    tl.store(accept_token_num_ptr + bx, num_accepted)

    # ------------------------------------------------------------------
    # Phase 2: Sample correction token from relu(target − draft)
    # ------------------------------------------------------------------
    coin_final = tl.load(uniform_final_ptr + bx).to(tl.float32)

    target_row_base = prob_batch_off + cur_prob_offset * vocab_size
    draft_row_base = prob_batch_off + cur_prob_offset * vocab_size

    # Two-pass approach over vocabulary:
    #   Pass 1 — compute sum of relu(target - draft)
    #   Pass 2 — running prefix sum to find first index where prefix > u

    sum_relu: tl.float32 = 0.0
    for v_start in tl.static_range(0, vocab_size, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        t_vals = tl.load(
            target_probs_ptr + target_row_base + offs, mask=mask, other=0.0
        ).to(tl.float32)
        d_vals = tl.load(
            draft_probs_ptr + draft_row_base + offs, mask=mask, other=0.0
        ).to(tl.float32)
        relu_vals = tl.maximum(t_vals - d_vals, 0.0)
        sum_relu += tl.sum(relu_vals)

    u = coin_final * sum_relu

    # Pass 2 — sequential block scan with running prefix
    running_prefix: tl.float32 = 0.0
    sampled_id: tl.int32 = vocab_size - 1  # default fallback

    found: tl.int32 = 0
    for v_start in tl.static_range(0, vocab_size, BLOCK_V):
        if found == 0:
            offs = v_start + tl.arange(0, BLOCK_V)
            mask = offs < vocab_size
            t_vals = tl.load(
                target_probs_ptr + target_row_base + offs, mask=mask, other=0.0
            ).to(tl.float32)
            d_vals = tl.load(
                draft_probs_ptr + draft_row_base + offs, mask=mask, other=0.0
            ).to(tl.float32)
            relu_vals = tl.maximum(t_vals - d_vals, 0.0)

            block_cumsum = tl.cumsum(relu_vals, axis=0)
            global_cumsum = block_cumsum + running_prefix

            # Mask: cumsum > u AND within valid range
            exceeds = (global_cumsum > u) & mask
            if tl.sum(exceeds.to(tl.int32)) > 0:
                # Find the minimum index where exceeds is True
                # Set non-exceeding positions to a large sentinel
                sentinel = vocab_size + 1
                candidate_ids = tl.where(exceeds, offs, sentinel)
                sampled_id = tl.min(candidate_ids).to(tl.int32)
                found = 1

            running_prefix += tl.sum(relu_vals)

    tl.store(predicts_ptr + last_accepted_retrive_idx, sampled_id)


def tree_speculative_sampling_target_only_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """Tree speculative sampling — Triton kernel, modifies tensors in-place."""
    bs = candidates.shape[0]
    num_draft_tokens = candidates.shape[1]
    num_spec_steps = accept_index.shape[1]
    vocab_size = target_probs.shape[2]

    # Choose block size: power-of-2, capped to avoid register pressure
    BLOCK_V = min(triton.next_power_of_2(vocab_size), 4096)

    _tree_spec_sampling_kernel[(bs,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        num_draft_tokens=num_draft_tokens,
        num_spec_steps=num_spec_steps,
        vocab_size=vocab_size,
        BLOCK_V=BLOCK_V,
    )
