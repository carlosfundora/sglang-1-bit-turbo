"""Medusa multi-head speculative decoding worker.

Generates K draft tokens in parallel using lightweight MLP heads on the target
model's last hidden state.  No autoregressive loop — all drafts from one pass.
Verification reuses the NgramVerifyInput tree infrastructure (same tree attention
as EAGLE3/NGRAM).

Worker pattern: coordinator (like NGRAMWorker / PCascadeWorker).
"""

import logging
import os
from typing import List, Optional

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Medusa tree choices (predefined candidate combinations from the paper)
# ---------------------------------------------------------------------------

# Simple linear chain: head-0 → head-1 → … → head-K
# Each entry is a "path" — len(path) = depth, path[-1] = top-k index
def _make_linear_choices(n_heads: int) -> List[List[int]]:
    return [[0] * (i + 1) for i in range(n_heads)]


# Standard Medusa mc_sim_7b_63 tree (63 candidates, 5 heads)
MC_SIM_63: List[List[int]] = [
    [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2],
    [3], [0, 0, 1], [1, 1], [0, 3], [4], [0, 0, 2], [1, 2],
    [2, 0], [0, 0, 3], [0, 0, 0, 0], [1, 0, 0], [0, 4],
    [1, 3], [2, 1], [3, 0], [0, 0, 4], [0, 0, 0, 1],
    [0, 1, 0], [1, 0, 1], [0, 0, 0, 2], [4, 0], [2, 2],
    [0, 1, 1], [3, 1], [1, 1, 0], [0, 2, 0], [1, 4],
    [0, 0, 0, 3], [2, 0, 0], [0, 3, 0], [0, 1, 2], [1, 0, 2],
    [0, 0, 1, 0], [3, 2], [2, 3], [4, 1], [1, 2, 0],
    [0, 0, 0, 4], [0, 2, 1], [1, 0, 0, 0], [0, 0, 0, 0, 0],
    [1, 1, 1], [0, 0, 2, 0], [0, 0, 1, 1], [2, 0, 1],
    [0, 1, 0, 0], [0, 4, 0], [3, 0, 0], [0, 1, 3],
    [1, 3, 0], [0, 0, 3, 0], [2, 1, 0], [4, 0, 0],
    [1, 0, 0, 1], [0, 3, 1],
]


def _build_linear_tree_mask(num_draft: int) -> np.ndarray:
    """Lower-triangular mask for a linear draft chain.

    Returns:
        (num_draft, num_draft) bool numpy array.
    """
    return np.tril(np.ones((num_draft, num_draft), dtype=bool))


# ---------------------------------------------------------------------------
# MedusaWorker
# ---------------------------------------------------------------------------


class MedusaWorker:
    """Coordinator that runs Medusa heads + NgramVerifyInput tree verification.

    The worker follows the same interface the scheduler expects (model_runner,
    model_config, forward_batch_generation, clear_cache_pool, …).

    Hidden-state capture:
        Medusa requires the target model's last hidden state.  After each
        extend or verification forward, we attempt to read it from
        ``model_runner.last_hidden_states``.  If unavailable, Medusa falls
        back to target-only decoding (no drafting) with a warning.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.target_worker = target_worker
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        # Medusa-specific config
        self.num_heads = getattr(server_args, "medusa_num_heads", 5)
        self.medusa_topk = getattr(server_args, "medusa_topk", 1)
        draft_tokens = server_args.speculative_num_draft_tokens or 8
        self.draft_token_num = min(draft_tokens, self.num_heads)

        # Load the model
        model_path = getattr(server_args, "medusa_model_path", None)
        if model_path is None:
            raise ValueError(
                "Medusa requires --medusa-model-path pointing to trained heads."
            )

        from sglang.srt.speculative.medusa_model import MedusaModel

        self.medusa_model = MedusaModel.from_pretrained(
            model_path,
            device=self.device,
            dtype=target_worker.model_runner.model_config.dtype,
        )
        logger.info(
            "MedusaWorker: loaded %d heads from %s", self.medusa_model.num_heads, model_path
        )

        # Use linear chain tree (simple, guaranteed to work)
        self.tree_mask_np = _build_linear_tree_mask(self.draft_token_num)

        # Pre-allocated tensors
        self.max_batch_size = target_worker.max_running_requests
        self._init_preallocated_tensors()

        # Hidden-state cache (populated after each target forward)
        self._cached_hidden: Optional[torch.Tensor] = None
        self._hidden_available = False

        logger.info(
            "MedusaWorker ready: %d heads, %d draft tokens, linear tree",
            self.num_heads,
            self.draft_token_num,
        )

    def _init_preallocated_tensors(self):
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )
        self._draft_tokens_buf = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self._positions_buf = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self._tree_mask_buf = torch.empty(
            (max_total_mask,), dtype=torch.bool, device=self.device
        )
        self._retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self._retrive_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self._retrive_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )

    # ---- Proxy attributes for scheduler ----

    @property
    def model_runner(self):
        return self.target_worker.model_runner

    @property
    def model_config(self):
        return self.target_worker.model_runner.model_config

    @property
    def max_running_requests(self):
        return self.target_worker.max_running_requests

    def get_memory_pool(self):
        return self.target_worker.get_memory_pool()

    def clear_cache_pool(self):
        self._cached_hidden = None
        self._hidden_available = False

    # ---- Main dispatch ----

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        # On extend, delegate to target
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            result = self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )
            self._try_capture_hidden()
            return result

        bs = batch.batch_size()
        if bs == 0 or batch.forward_mode.is_idle():
            return self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )

        # Try to generate Medusa drafts
        if self._hidden_available and self._cached_hidden is not None:
            try:
                return self._forward_medusa(batch)
            except Exception as e:
                logger.warning("Medusa draft failed (%s), falling back to target-only", e)

        # Fallback: target-only decode
        result = self.target_worker.forward_batch_generation(
            batch.get_model_worker_batch()
        )
        self._try_capture_hidden()
        return result

    # ---- Medusa draft + verify ----

    def _forward_medusa(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Generate Medusa drafts and verify via NgramVerifyInput."""
        from sglang.srt.speculative.ngram_info import NgramVerifyInput

        bs = batch.batch_size()
        K = self.draft_token_num

        # Step 1: Run Medusa heads on cached hidden states
        hidden = self._cached_hidden
        if hidden.shape[0] < bs:
            # Pad or truncate to batch size
            hidden = hidden[:bs]
        elif hidden.shape[0] > bs:
            hidden = hidden[:bs]

        with torch.no_grad():
            draft_tokens = self.medusa_model.predict_tokens(hidden)  # [bs, num_heads]
            draft_tokens = draft_tokens[:, :K]  # trim to draft_token_num

        # Step 2: Build flat draft token + tree mask arrays (ngram format)
        flat_drafts = draft_tokens.reshape(-1).contiguous()  # [bs * K]

        # Tree mask: tile the linear mask for each request
        mask_per_req = self.tree_mask_np.flatten()  # K*K bools
        mask_tiled = np.tile(mask_per_req, bs)

        # Copy to pre-allocated GPU tensors
        self._draft_tokens_buf[: bs * K].copy_(flat_drafts, non_blocking=True)
        self._tree_mask_buf[: bs * K * K].copy_(
            torch.from_numpy(mask_tiled), non_blocking=True
        )

        # Step 3: Reconstruct tree indices
        positions = self._positions_buf[: bs * K]
        retrive_index = self._retrieve_indexes[:bs, :K]
        retrive_next_token = self._retrive_next_token[:bs, :K]
        retrive_next_sibling = self._retrive_next_sibling[:bs, :K]
        tree_mask_gpu = self._tree_mask_buf[: bs * K * K]

        try:
            from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

            reconstruct_indices_from_tree_mask(
                tree_mask_gpu,
                batch.seq_lens,
                positions,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                bs,
                K,
            )
        except Exception as e:
            logger.warning("Medusa reconstruct_indices failed (%s), falling back", e)
            return self._fallback_target_only(batch)

        # Step 4: Build full attention mask (prefix + tree)
        tree_mask_list = []
        mask_np_3d = mask_tiled.reshape(bs, K, K)
        for i, req in enumerate(batch.reqs):
            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            prefix_mask = torch.ones((K, seq_len - 1), device="cuda")
            tree_part = torch.from_numpy(mask_np_3d[i]).cuda()
            full_mask = torch.cat((prefix_mask, tree_part), dim=1).to(torch.bool)
            tree_mask_list.append(full_mask.flatten())
        full_tree_mask = torch.cat(tree_mask_list, dim=0)

        # Step 5: Set verify mode and run target
        original_algo = batch.spec_algorithm
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM  # reuse ngram verify path
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = NgramVerifyInput(
            self._draft_tokens_buf[: bs * K],
            full_tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            K,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

        model_worker_batch = batch.get_model_worker_batch()

        if model_worker_batch.forward_mode.is_target_verify():
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            logits_output = batch_result.logits_output
            can_run_cuda_graph = batch_result.can_run_cuda_graph

            verify_input: NgramVerifyInput = model_worker_batch.spec_info
            logits_output, next_token_ids, num_accepted = verify_input.verify(
                batch, logits_output, self.page_size, None
            )
            accept_lens = verify_input.accept_length
            batch.forward_mode = ForwardMode.DECODE
            batch.spec_algorithm = original_algo

            # Capture hidden states for next round
            self._try_capture_hidden()

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=num_accepted,
                can_run_cuda_graph=can_run_cuda_graph,
                accept_lens=accept_lens,
            )

        # Unexpected mode — fallback
        batch.spec_algorithm = original_algo
        return self._fallback_target_only(batch)

    # ---- Hidden state capture ----

    def _try_capture_hidden(self):
        """Try to read last hidden states from the target model runner."""
        try:
            mr = self.target_worker.model_runner
            h = getattr(mr, "last_hidden_states", None)
            if h is not None and isinstance(h, torch.Tensor) and h.numel() > 0:
                self._cached_hidden = h.detach()
                self._hidden_available = True
                return
        except Exception:
            pass
        self._hidden_available = False

    # ---- Fallback ----

    def _fallback_target_only(self, batch: ScheduleBatch) -> GenerationBatchResult:
        batch.forward_mode = ForwardMode.DECODE
        result = self.target_worker.forward_batch_generation(
            batch.get_model_worker_batch()
        )
        self._try_capture_hidden()
        return result
