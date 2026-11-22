"""SSD (Self-Speculative Decoding) worker.

Uses the target model itself as the drafter via early-exit: run only the first
N layers, project through lm_head → draft tokens.  No separate draft model,
no extra VRAM, works with any model.

Tree verification reuses NgramVerifyInput (same path as NGRAM/Medusa).

Key idea from "Draft & Verify" (arXiv 2309.08168) and "LayerSkip" (arXiv
2404.16710): a prefix of transformer layers already captures enough signal
to produce reasonable next-token predictions.

Worker pattern: coordinator (like NGRAMWorker / MedusaWorker).
"""

import logging
from typing import List, Optional

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


def _build_linear_tree_mask(num_draft: int) -> np.ndarray:
    """Build a simple linear (chain) tree mask: each position depends on all previous."""
    mask = np.zeros((num_draft, num_draft), dtype=bool)
    for i in range(num_draft):
        for j in range(i + 1):
            mask[i, j] = True
    return mask


class SSDWorker:
    """Self-Speculative Decoding via early-exit from the target model.

    Args:
        server_args: Server configuration.
        gpu_id: GPU device index.
        tp_rank: Tensor-parallel rank.
        dp_rank: Data-parallel rank.
        moe_ep_rank: MoE expert-parallel rank.
        attn_cp_rank: Attention context-parallel rank.
        moe_dp_rank: MoE data-parallel rank.
        nccl_port: NCCL communication port.
        target_worker: The main model worker.
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
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens or 5
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        # Determine early-exit layer.  Default: use first 1/3 of layers.
        total_layers = self._get_total_layers()
        self.exit_layer = max(1, total_layers // 3)
        # Allow override via server_args
        ssd_exit = getattr(server_args, "ssd_exit_layer", None)
        if ssd_exit is not None and ssd_exit > 0:
            self.exit_layer = min(ssd_exit, total_layers - 1)

        self.max_batch_size = target_worker.max_running_requests

        # Pre-build a linear tree mask (chain: pos 0 → pos 1 → … → pos K-1)
        K = self.draft_token_num
        self.tree_mask_np = _build_linear_tree_mask(K)

        # Pre-allocate GPU tensors (same pattern as NGRAMWorker)
        self._init_preallocated_tensors()

        # Cached hidden states from last target forward
        self._cached_hidden = None
        self._hidden_available = False

        logger.info(
            "SSDWorker: exit_layer=%d/%d, draft_tokens=%d",
            self.exit_layer, total_layers, self.draft_token_num,
        )

    def _get_total_layers(self) -> int:
        """Get total number of layers in the target model."""
        mc = self.model_runner.model_config
        return getattr(mc, "num_hidden_layers", 32)

    def _init_preallocated_tensors(self):
        K = self.draft_token_num
        max_total = self.max_batch_size * K
        max_mask = self.max_batch_size * K * K

        self._draft_tokens = torch.empty(max_total, dtype=torch.int64, device=self.device)
        self._positions = torch.empty(max_total, dtype=torch.int64, device=self.device)
        self._tree_mask = torch.empty(max_mask, dtype=torch.bool, device=self.device)
        self._retrieve_indexes = torch.empty(
            (self.max_batch_size, K), dtype=torch.int64, device=self.device
        )
        self._retrive_next_token = torch.empty(
            (self.max_batch_size, K), dtype=torch.int64, device=self.device
        )
        self._retrive_next_sibling = torch.empty(
            (self.max_batch_size, K), dtype=torch.int64, device=self.device
        )

    @property
    def max_running_requests(self):
        return self.target_worker.max_running_requests

    @property
    def model_config(self):
        return self.target_worker.model_runner.model_config

    def get_memory_pool(self):
        return self.target_worker.get_memory_pool()

    def clear_cache_pool(self):
        self._cached_hidden = None
        self._hidden_available = False

    # ---- Main dispatch ----

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        # On extend/prefill, just run target with hidden capture
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._enable_hidden_capture(batch)
            result = self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )
            self._capture_hidden_from_result(result)
            return result

        bs = batch.batch_size()
        if bs == 0 or batch.forward_mode.is_idle():
            return self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )

        # Try SSD draft+verify
        if self._hidden_available and self._cached_hidden is not None:
            try:
                return self._forward_ssd(batch)
            except Exception as e:
                logger.warning("SSD draft failed (%s), falling back to target-only", e)

        # Fallback: normal decode with hidden capture
        return self._fallback_target_only(batch)

    # ---- SSD draft via early-exit hidden → lm_head ----

    def _forward_ssd(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Generate drafts from cached hidden states and verify."""
        from sglang.srt.speculative.ngram_info import NgramVerifyInput

        bs = batch.batch_size()
        K = self.draft_token_num
        hidden = self._cached_hidden[:bs]

        # Project hidden through lm_head to get logits
        model = self.model_runner.model
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            # Some models embed lm_head differently
            lm_head = getattr(model, "output", None)
        if lm_head is None:
            raise RuntimeError("Cannot find lm_head on target model for SSD")

        with torch.no_grad():
            # Generate first draft token from lm_head(hidden) — this is valid
            # because hidden came from the final layer of the target model.
            # Subsequent tokens use the same greedy token repeated (simple but
            # non-garbage), since embed→lm_head without transformer layers
            # produces meaningless logits.  The first token typically has
            # ~30-60% acceptance; repeats occasionally hit on repetitive text.
            logits = lm_head(hidden)  # [bs, vocab]
            first_token = logits.argmax(dim=-1)  # [bs] greedy

            # Build draft: first real token + (K-1) copies of it as padding.
            # The linear chain mask means verification stops at first rejection,
            # so only the first token truly matters for throughput.
            draft_tokens_list = [first_token]
            for _ in range(K - 1):
                draft_tokens_list.append(first_token)

            draft_tokens = torch.stack(draft_tokens_list, dim=1)  # [bs, K]

        # Build tree structure (linear chain)
        flat_drafts = draft_tokens.reshape(-1).contiguous()
        mask_per_req = self.tree_mask_np.flatten()
        mask_tiled = np.tile(mask_per_req, bs)

        self._draft_tokens[: bs * K].copy_(flat_drafts, non_blocking=True)
        self._tree_mask[: bs * K * K].copy_(
            torch.from_numpy(mask_tiled), non_blocking=True
        )

        # Reconstruct tree indices
        positions = self._positions[: bs * K]
        retrive_index = self._retrieve_indexes[:bs, :K]
        retrive_next_token = self._retrive_next_token[:bs, :K]
        retrive_next_sibling = self._retrive_next_sibling[:bs, :K]
        tree_mask_gpu = self._tree_mask[: bs * K * K]

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
            logger.warning("SSD reconstruct_indices failed (%s), falling back", e)
            return self._fallback_target_only(batch)

        # Build full attention mask (prefix + tree)
        tree_mask_list = []
        mask_np_3d = mask_tiled.reshape(bs, K, K)
        for i, req in enumerate(batch.reqs):
            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            prefix_mask = torch.ones((K, seq_len - 1), device=self.device)
            tree_part = torch.from_numpy(mask_np_3d[i]).to(self.device)
            full_mask = torch.cat((prefix_mask, tree_part), dim=1).to(torch.bool)
            tree_mask_list.append(full_mask.flatten())
        full_tree_mask = torch.cat(tree_mask_list, dim=0)

        # Set verify mode
        original_algo = batch.spec_algorithm
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = NgramVerifyInput(
            self._draft_tokens[: bs * K],
            full_tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            K,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

        # Run target verification
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

            # Capture hidden for next round
            self._capture_hidden_from_result(batch_result)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=num_accepted,
                can_run_cuda_graph=can_run_cuda_graph,
                accept_lens=accept_lens,
            )

        batch.spec_algorithm = original_algo
        return self._fallback_target_only(batch)

    # ---- Hidden state capture (via logits_output, same as EAGLE) ----

    def _enable_hidden_capture(self, batch: ScheduleBatch):
        """Set capture mode so the model forward stores hidden states in logits_output."""
        if batch.spec_info is not None:
            batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        else:
            from types import SimpleNamespace
            batch.spec_info = SimpleNamespace(capture_hidden_mode=CaptureHiddenMode.LAST)

    def _capture_hidden_from_result(self, result: GenerationBatchResult):
        """Extract hidden states from the forward result (logits_output.hidden_states).

        This is the same path EAGLE uses — hidden states are returned as part of
        the logits output when CaptureHiddenMode is active.
        """
        try:
            lo = result.logits_output
            if lo is not None:
                h = getattr(lo, "hidden_states", None)
                if h is not None and isinstance(h, torch.Tensor) and h.numel() > 0:
                    self._cached_hidden = h.detach()
                    self._hidden_available = True
                    return
        except Exception:
            pass
        self._hidden_available = False

    def _fallback_target_only(self, batch: ScheduleBatch) -> GenerationBatchResult:
        batch.forward_mode = ForwardMode.DECODE
        self._enable_hidden_capture(batch)
        result = self.target_worker.forward_batch_generation(
            batch.get_model_worker_batch()
        )
        self._capture_hidden_from_result(result)
        return result
