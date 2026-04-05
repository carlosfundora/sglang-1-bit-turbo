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
"""Pure PyTorch fallback for tree_speculative_sampling_target_only.

Semantically equivalent to the CUDA/HIP C++ kernel. Intended as the
universal Tier-3 fallback that works on any device PyTorch supports.
"""

from __future__ import annotations

import torch


def tree_speculative_sampling_target_only_pytorch(
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
    """Tree speculative sampling — pure PyTorch, modifies tensors in-place.

    Semantics match the CUDA ``tree_speculative_sampling_target_only`` kernel:
      Phase 1 — walk the draft tree, probabilistically accepting tokens.
      Phase 2 — sample a correction token from relu(target - draft).
    """
    bs = candidates.shape[0]
    num_draft_tokens = candidates.shape[1]
    num_spec_steps = accept_index.shape[1]
    vocab_size = target_probs.shape[2]

    # Pre-copy small tree-structure tensors to CPU to avoid per-element syncs.
    candidates_cpu = candidates.cpu().tolist()
    retrive_index_cpu = retrive_index.cpu().tolist()
    retrive_next_token_cpu = retrive_next_token.cpu().tolist()
    retrive_next_sibling_cpu = retrive_next_sibling.cpu().tolist()
    uniform_samples_cpu = uniform_samples.cpu().tolist()
    uniform_final_cpu = uniform_samples_for_final_sampling.cpu().tolist()

    for bx in range(bs):
        # ------------------------------------------------------------------
        # Phase 1: Tree walk with probabilistic acceptance
        # ------------------------------------------------------------------
        prob_acc = 0.0
        cur_prob_offset = 0  # index within this batch's draft tokens
        coin = uniform_samples_cpu[bx][0]
        last_accepted_retrive_idx = retrive_index_cpu[bx][0]
        accept_index[bx, 0] = last_accepted_retrive_idx
        num_accepted = 0
        cur_index = 0

        for _j in range(1, num_spec_steps):
            cur_index = retrive_next_token_cpu[bx][cur_index]

            while cur_index != -1:
                draft_index = retrive_index_cpu[bx][cur_index]
                draft_token_id = candidates_cpu[bx][cur_index]

                target_prob_single = target_probs[
                    bx, cur_prob_offset, draft_token_id
                ].item()
                prob_acc += target_prob_single

                threshold_acc_safe = max(threshold_acc, 1e-9)
                if (
                    coin <= prob_acc / threshold_acc_safe
                    or target_prob_single >= threshold_single
                ):
                    # ACCEPT
                    prob_acc = 0.0
                    cur_prob_offset = cur_index
                    coin = uniform_samples_cpu[bx][cur_index]
                    predicts[last_accepted_retrive_idx] = draft_token_id
                    num_accepted += 1
                    accept_index[bx, num_accepted] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break
                else:
                    # REJECT — record target mass for correction step
                    draft_probs[bx, cur_prob_offset, draft_token_id] = (
                        target_probs[bx, cur_prob_offset, draft_token_id]
                    )
                    cur_index = retrive_next_sibling_cpu[bx][cur_index]

            if cur_index == -1:
                break

        accept_token_num[bx] = num_accepted

        # ------------------------------------------------------------------
        # Phase 2: Sample correction token from relu(target − draft)
        # ------------------------------------------------------------------
        coin_final = uniform_final_cpu[bx]

        target_row = target_probs[bx, cur_prob_offset]  # [vocab_size]
        draft_row = draft_probs[bx, cur_prob_offset]  # [vocab_size]
        relu_diff = (target_row - draft_row).clamp(min=0.0)

        sum_relu = relu_diff.sum().item()
        u = coin_final * sum_relu

        if sum_relu > 0.0:
            cumsum = relu_diff.cumsum(dim=0)
            # First index where cumsum > u (matches kernel: vocab_size - 1 fallback)
            mask = cumsum > u
            if mask.any():
                sampled_id = mask.to(torch.long).argmax().item()
            else:
                sampled_id = vocab_size - 1
        else:
            # No positive mass — kernel falls through to vocab_size - 1
            sampled_id = vocab_size - 1

        predicts[last_accepted_retrive_idx] = sampled_id
