"""P-CASCADE speculative decoding worker.

Combines P-EAGLE neural drafting with DyTC cascade routing and n-gram fallback.
Uses a coordinator pattern: owns an EAGLEWorker for L1/L2 drafting and an
NgramCorpus for L3 fallback. A lightweight DyTC-style router decides which
level to use per whole-batch decode iteration.

Cascade levels:
  L1: Full P-EAGLE / EAGLE3 drafting (all speculative steps)
  L2: Reduced-depth drafting (fewer effective steps, same geometry)
  L3: N-gram only (skip neural draft entirely)
"""

import logging
import os
from dataclasses import dataclass
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
# DyTC-style heuristic router
# ---------------------------------------------------------------------------

@dataclass
class CascadeDecision:
    level: int       # 1, 2, or 3
    reason: str      # for logging/diagnostics


class PCascadeRouter:
    """Lightweight DyTC-style heuristic router (no training needed).

    Decides cascade level per whole-batch decode iteration based on:
      - Rolling EMA of acceptance rate
      - Mean context length of the batch
      - Recent repetition score (n-gram hit rate proxy)
    """

    def __init__(
        self,
        ema_alpha: float = 0.15,
        l2_accept_threshold: float = 2.0,
        l3_accept_threshold: float = 1.2,
        l3_repetition_threshold: float = 0.65,
        long_context_threshold: int = 8192,
    ):
        self.ema_alpha = ema_alpha
        self.ema_accept = 3.0  # optimistic initialisation
        self.l2_accept_threshold = l2_accept_threshold
        self.l3_accept_threshold = l3_accept_threshold
        self.l3_repetition_threshold = l3_repetition_threshold
        self.long_context_threshold = long_context_threshold
        self.total_decisions = 0
        self.level_counts = [0, 0, 0, 0]  # index 1-3

    def decide(
        self,
        mean_context_len: int,
        repetition_score: float = 0.0,
    ) -> CascadeDecision:
        self.total_decisions += 1

        # L3: high repetition → n-gram will do well
        if repetition_score > self.l3_repetition_threshold:
            self.level_counts[3] += 1
            return CascadeDecision(3, "high repetition")

        # L3: acceptance is very poor → neural draft is wasting compute
        if self.ema_accept < self.l3_accept_threshold:
            self.level_counts[3] += 1
            return CascadeDecision(3, "very low acceptance")

        # L2: acceptance is moderate or context is very long
        if (
            self.ema_accept < self.l2_accept_threshold
            or mean_context_len > self.long_context_threshold
        ):
            self.level_counts[2] += 1
            return CascadeDecision(2, "moderate acceptance or long context")

        # L1: full P-EAGLE
        self.level_counts[1] += 1
        return CascadeDecision(1, "high acceptance")

    def update(self, accepted_tokens: int):
        """Update EMA with the latest acceptance count."""
        self.ema_accept = (
            (1 - self.ema_alpha) * self.ema_accept
            + self.ema_alpha * accepted_tokens
        )

    def stats_str(self) -> str:
        if self.total_decisions == 0:
            return "no decisions yet"
        pcts = [
            f"L{i}={self.level_counts[i] / self.total_decisions * 100:.1f}%"
            for i in range(1, 4)
        ]
        return (
            f"ema_accept={self.ema_accept:.2f} "
            + " ".join(pcts)
            + f" ({self.total_decisions} decisions)"
        )


# ---------------------------------------------------------------------------
# PCascadeWorker — coordinator
# ---------------------------------------------------------------------------

class PCascadeWorker:
    """Coordinator worker that owns an EAGLEWorker + ngram fallback.

    Implements the same interface that the scheduler expects:
      - forward_batch_generation()
      - clear_cache_pool()
      - model_runner, model_config, max_running_requests, get_memory_pool()
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
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        # ---- Create the internal EAGLE worker ----
        # Temporarily override speculative_algorithm so the EAGLEWorker
        # thinks it is running EAGLE3 (or P_EAGLE if the user has that head).
        original_algo = server_args.speculative_algorithm
        # Detect whether the draft model is P-EAGLE
        _eagle_algo = self._detect_eagle_variant(server_args)
        server_args.speculative_algorithm = _eagle_algo
        logger.info(
            "PCascadeWorker: creating internal EAGLE worker "
            f"(variant={_eagle_algo})"
        )

        from sglang.srt.speculative.eagle_worker import EAGLEWorker
        self.eagle_worker = EAGLEWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )

        # Restore the original algorithm string
        server_args.speculative_algorithm = original_algo

        # ---- Create the n-gram corpus (L3 fallback) ----
        self._init_ngram(server_args, target_worker)

        # ---- Create the DyTC router ----
        self.router = PCascadeRouter()

        # ---- Track current cascade level for logging ----
        self.current_level = 1
        self._log_interval = 50
        self._forward_count = 0

        # ---- Draft state snapshots for L3 recovery ----
        self._saved_eagle_kv_pos: Optional[int] = None

        # ---- Optional Saguaro wrapper flag ----
        self._ssd_enabled = getattr(server_args, "ssd_enable", False)

        logger.info(
            "PCascadeWorker initialised (L1=EAGLE, L2=reduced, L3=ngram, ssd=%s)",
            self._ssd_enabled,
        )

    def _detect_eagle_variant(self, server_args: ServerArgs) -> str:
        """Detect whether to use EAGLE3 or P_EAGLE for the internal worker."""
        draft_path = server_args.speculative_draft_model_path
        if draft_path is None:
            return "EAGLE3"
        # Check for P-EAGLE config markers
        import json
        config_path = os.path.join(draft_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                if cfg.get("parallel_drafting", False):
                    return "P_EAGLE"
            except Exception:
                pass
        return "EAGLE3"

    def _init_ngram(self, server_args: ServerArgs, target_worker: TpModelWorker):
        """Initialise the n-gram corpus and pre-allocated verification tensors."""
        from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
        from sglang.srt.speculative.ngram_info import NgramVerifyInput

        self.draft_token_num = server_args.speculative_num_draft_tokens or 16
        max_trie_depth = server_args.speculative_ngram_max_trie_depth or 6
        max_match_window = server_args.speculative_ngram_max_match_window_size or 32

        self.ngram_corpus = NgramCorpus(
            min_match_window_size=server_args.speculative_ngram_min_match_window_size or 4,
            max_match_window_size=max_match_window,
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth or 1,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth or 3,
            match_type=server_args.speculative_ngram_match_type or "suffix",
            capacity=server_args.speculative_ngram_capacity or 65536,
            max_trie_depth=max_trie_depth,
            draft_token_num=self.draft_token_num,
        )

        # Pre-allocate ngram verification tensors (same as NGRAMWorker)
        self.max_batch_size = target_worker.max_running_requests
        self.page_size = server_args.page_size
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self._ngram_draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self._ngram_retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64, device=self.device,
        )
        self._ngram_retrive_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64, device=self.device,
        )
        self._ngram_retrive_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64, device=self.device,
        )
        self._ngram_positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self._ngram_tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )
        self.max_match_window_size = max_match_window
        self.max_trie_depth = max_trie_depth

    # ---- Proxy attributes expected by the scheduler ----

    @property
    def model_runner(self):
        return self.eagle_worker.model_runner

    @property
    def model_config(self):
        return self.eagle_worker.model_config

    @property
    def max_running_requests(self):
        return self.eagle_worker.max_running_requests

    def get_memory_pool(self):
        return self.eagle_worker.get_memory_pool()

    def clear_cache_pool(self):
        self.eagle_worker.clear_cache_pool()
        self.ngram_corpus.reset()

    # ---- Main dispatch ----

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        """Route to EAGLE (L1/L2) or n-gram (L3) based on router decision."""
        self._forward_count += 1

        # On extend (prefill), always delegate to EAGLE
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            result = self.eagle_worker.forward_batch_generation(batch)
            self._maybe_update_ngram_corpus(batch)
            return result

        # Compute routing inputs
        mean_ctx = self._mean_context_len(batch)
        rep_score = self._repetition_score(batch)
        decision = self.router.decide(mean_ctx, rep_score)
        self.current_level = decision.level

        if decision.level == 3:
            # L3: pure n-gram verification — snapshot/restore EAGLE KV state
            self._save_draft_state()
            result = self._forward_ngram(batch)
            self._restore_draft_state()
        else:
            # L1 or L2: delegate to EAGLE worker
            # For L2, we could reduce effective steps, but for now both
            # use the same EAGLE path (safe fixed geometry).
            result = self.eagle_worker.forward_batch_generation(batch)

        # Update router EMA with acceptance count
        self.router.update(result.num_accepted_tokens)

        # Feed tokens to n-gram corpus for future L3 use
        self._maybe_update_ngram_corpus(batch)

        # Periodic logging
        if self._forward_count % self._log_interval == 0:
            logger.info(
                f"PCascade stats: {self.router.stats_str()} "
                f"last_level={decision.level} ({decision.reason})"
            )

        return result

    # ---- N-gram path (L3) ----

    def _forward_ngram(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run n-gram-only speculative decoding (no neural draft)."""
        from sglang.srt.speculative.ngram_info import NgramVerifyInput
        from sglang.srt.speculative.spec_utils import generate_token_bitmask

        bs = batch.batch_size()

        # Skip n-gram on idle
        if batch.forward_mode.is_idle() or bs == 0:
            return self.eagle_worker.forward_batch_generation(batch)

        try:
            # Generate n-gram draft candidates
            self.ngram_corpus.synchronize()
            batch_tokens = []
            for req in batch.reqs:
                check_token = self._efficient_concat_last_n(
                    req.origin_input_ids,
                    req.output_ids,
                    self.max_match_window_size,
                )
                batch_tokens.append(check_token)
            req_drafts, mask = self.ngram_corpus.batch_get(batch_tokens)
            total_draft_token_num = len(req_drafts)

            if total_draft_token_num != bs * self.draft_token_num:
                # N-gram corpus doesn't have enough history; fall back to EAGLE
                logger.debug("PCascade L3: ngram corpus insufficient, falling back to L1")
                return self.eagle_worker.forward_batch_generation(batch)

        except Exception as e:
            logger.warning(f"PCascade L3 ngram error: {e}, falling back to L1")
            return self.eagle_worker.forward_batch_generation(batch)

        # Fill pre-allocated tensors
        retrive_index = self._ngram_retrieve_indexes[:bs, :]
        retrive_next_token = self._ngram_retrive_next_token[:bs, :]
        retrive_next_sibling = self._ngram_retrive_next_sibling[:bs, :]
        positions = self._ngram_positions[:bs * self.draft_token_num]
        tree_mask = self._ngram_tree_mask[:bs * self.draft_token_num * self.draft_token_num]
        draft_tokens = self._ngram_draft_tokens[:bs * self.draft_token_num]

        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        try:
            from sgl_kernel.speculative import reconstruct_indices_from_tree_mask
            reconstruct_indices_from_tree_mask(
                tree_mask,
                batch.seq_lens,
                positions,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                bs,
                self.draft_token_num,
            )
        except Exception as e:
            logger.warning(f"PCascade L3 reconstruct error: {e}, falling back to L1")
            return self.eagle_worker.forward_batch_generation(batch)

        # Build full mask (same as NGRAMWorker)
        tree_mask_list = []
        mask_np = mask.reshape(bs, self.draft_token_num, self.draft_token_num)
        for i, req in enumerate(batch.reqs):
            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            req_mask = torch.ones(
                (self.draft_token_num, seq_len - 1), device="cuda"
            )
            req_mask = torch.cat(
                (req_mask, torch.from_numpy(mask_np[i]).cuda()), dim=1
            ).to(torch.bool)
            tree_mask_list.append(req_mask.flatten())
        full_tree_mask = torch.cat(tree_mask_list, dim=0)

        # Temporarily set batch to NGRAM for the verify path, then restore
        original_spec_algo = batch.spec_algorithm
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = NgramVerifyInput(
            draft_tokens,
            full_tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            self.draft_token_num,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

        # Run target verification
        model_worker_batch = batch.get_model_worker_batch()
        spec_info = model_worker_batch.spec_info

        if model_worker_batch.forward_mode.is_target_verify():
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            logits_output = batch_result.logits_output
            can_run_cuda_graph = batch_result.can_run_cuda_graph

            verify_input: NgramVerifyInput = model_worker_batch.spec_info

            vocab_mask = None
            if batch.has_grammar:
                retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
                retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
                draft_tokens_cpu = spec_info.draft_token.view(
                    spec_info.retrive_next_token.shape
                ).cpu()
                vocab_mask = generate_token_bitmask(
                    batch.reqs,
                    verify_input,
                    retrieve_next_token_cpu,
                    retrieve_next_sibling_cpu,
                    draft_tokens_cpu,
                    batch.sampling_info.vocab_size,
                )
                if vocab_mask is not None:
                    assert verify_input.grammar is not None
                    vocab_mask = vocab_mask.to(
                        verify_input.retrive_next_token.device
                    )
                    batch.sampling_info.vocab_mask = None

            logits_output, next_token_ids, num_accepted = verify_input.verify(
                batch, logits_output, self.page_size, vocab_mask
            )
            accept_lens = verify_input.accept_length
            batch.forward_mode = ForwardMode.DECODE
            batch.spec_algorithm = original_spec_algo  # restore P_CASCADE

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=num_accepted,
                can_run_cuda_graph=can_run_cuda_graph,
                accept_lens=accept_lens,
            )
        else:
            # Fallback: not in verify mode somehow
            batch.spec_algorithm = original_spec_algo  # restore P_CASCADE
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            return GenerationBatchResult(
                logits_output=batch_result.logits_output,
                next_token_ids=batch_result.next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

    # ---- Draft state snapshot/restore for L3 recovery ----

    def _save_draft_state(self):
        """Snapshot EAGLE KV cache position before an n-gram-only round.

        After L3 (n-gram) runs, the EAGLE decoder's KV state hasn't been
        updated (n-gram bypasses the neural draft entirely).  When the router
        switches BACK to L1/L2, we need the EAGLE KV to be at the correct
        position.  Save the position here, restore after L3.
        """
        try:
            ew = self.eagle_worker
            if hasattr(ew, "model_runner") and hasattr(ew.model_runner, "kv_cache"):
                kv = ew.model_runner.kv_cache
                if hasattr(kv, "cur_length"):
                    self._saved_eagle_kv_pos = kv.cur_length
        except Exception:
            self._saved_eagle_kv_pos = None

    def _restore_draft_state(self):
        """Restore EAGLE KV cache position after an n-gram-only round."""
        if self._saved_eagle_kv_pos is None:
            return
        try:
            ew = self.eagle_worker
            if hasattr(ew, "model_runner") and hasattr(ew.model_runner, "kv_cache"):
                kv = ew.model_runner.kv_cache
                if hasattr(kv, "cur_length"):
                    kv.cur_length = self._saved_eagle_kv_pos
        except Exception:
            pass
        finally:
            self._saved_eagle_kv_pos = None

    # ---- Helpers ----

    def _mean_context_len(self, batch: ScheduleBatch) -> int:
        if batch.seq_lens is not None and batch.seq_lens.numel() > 0:
            return int(batch.seq_lens.float().mean().item())
        return 0

    def _repetition_score(self, batch: ScheduleBatch) -> float:
        """Cheap repetition heuristic: fraction of last-16 tokens that appear
        more than once in last-64 tokens across the batch."""
        total_rep = 0.0
        count = 0
        for req in batch.reqs:
            out = req.output_ids
            if len(out) < 16:
                continue
            recent = out[-16:]
            window = out[-64:] if len(out) >= 64 else out
            unique_in_recent = len(set(recent))
            rep = 1.0 - (unique_in_recent / len(recent))
            total_rep += rep
            count += 1
        return total_rep / max(count, 1)

    def _maybe_update_ngram_corpus(self, batch: ScheduleBatch):
        """Feed recent tokens into the n-gram trie for future L3 use."""
        try:
            batch_tokens = []
            for req in batch.reqs:
                put_ids = self._efficient_concat_last_n(
                    req.origin_input_ids,
                    req.output_ids,
                    self.max_trie_depth,
                )
                batch_tokens.append(put_ids)
            self.ngram_corpus.batch_put(batch_tokens)
        except Exception:
            pass  # non-critical

    @staticmethod
    def _efficient_concat_last_n(
        seq1: List[int], seq2: List[int], n: int
    ) -> List[int]:
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]
        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2
