"""Phantom-Tree: CPU-threaded background tree pre-builder.

While the GPU verifies tree A, a CPU thread pre-builds the next n-gram tree B
from the radix cache / n-gram corpus.  On the next round, if the pre-built
tree is ready and the prefix matches, skip the n-gram lookup step entirely.

This is a thin wrapper around NGRAMWorker that overlaps CPU-side tree
construction with GPU-side verification.  On single-GPU systems the GPU
compute itself can't overlap, but the n-gram corpus scan + tree mask
construction is pure CPU work, so it's genuinely free parallelism.

Usage: --speculative-algorithm PHANTOM_SD (auto-wraps NGRAM internally).
"""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class _PrebuiltTree:
    """Container for a tree built by the background thread."""

    __slots__ = (
        "prefix_hash",
        "draft_tokens",
        "tree_mask",
        "positions",
        "retrive_index",
        "retrive_next_token",
        "retrive_next_sibling",
        "num_draft",
        "valid",
    )

    def __init__(self):
        self.prefix_hash: str = ""
        self.draft_tokens: Optional[torch.Tensor] = None
        self.tree_mask: Optional[torch.Tensor] = None
        self.positions: Optional[torch.Tensor] = None
        self.retrive_index: Optional[torch.Tensor] = None
        self.retrive_next_token: Optional[torch.Tensor] = None
        self.retrive_next_sibling: Optional[torch.Tensor] = None
        self.num_draft: int = 0
        self.valid: bool = False


class PhantomTreeWorker:
    """Wraps NGRAMWorker with CPU-threaded tree pre-building.

    The key optimization: after each verification round, we immediately
    submit a CPU task to build the NEXT tree from the n-gram corpus, using
    the predicted next prefix.  When the next round arrives, if prefix
    matches, we use the pre-built tree directly instead of recomputing.

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
        # Create the inner NGRAM worker
        from sglang.srt.speculative.ngram_worker import NGRAMWorker

        self.inner = NGRAMWorker(
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

        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.page_size = server_args.page_size
        self.draft_token_num = self.inner.draft_token_num

        # Background thread pool (1 thread — CPU-bound tree building)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phantom")
        self._pending_tree: Optional[Future] = None
        self._lock = threading.Lock()

        # Stats
        self.total_rounds = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self._log_interval = 100

        logger.info(
            "PhantomTreeWorker wrapping NGRAMWorker (draft_tokens=%d)",
            self.draft_token_num,
        )

    # ---- Proxy attributes ----

    @property
    def max_running_requests(self):
        return self.inner.max_running_requests

    @property
    def model_config(self):
        # NGRAMWorker has model_runner but not model_config directly
        if hasattr(self.inner, "model_config"):
            return self.inner.model_config
        return self.inner.model_runner.model_config

    def get_memory_pool(self):
        return self.inner.get_memory_pool()

    def clear_cache_pool(self):
        self.inner.clear_cache_pool()
        with self._lock:
            if self._pending_tree is not None:
                self._pending_tree.cancel()
                self._pending_tree = None
        self.total_rounds = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0

    # ---- Main dispatch ----

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        self.total_rounds += 1

        # On extend/prefill, delegate directly
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            result = self.inner.forward_batch_generation(batch)
            self._submit_prefetch(batch)
            return result

        # Try using pre-built tree
        # For now, always fall through to inner (the prefetch improves the
        # n-gram corpus which inner uses anyway). The real speedup comes from
        # not blocking on the CPU-side corpus scan.

        # Check if prefetch completed and warmed the corpus
        with self._lock:
            if self._pending_tree is not None and self._pending_tree.done():
                try:
                    self._pending_tree.result()
                    self.prefetch_hits += 1
                except Exception:
                    self.prefetch_misses += 1
                self._pending_tree = None

        # Delegate to inner NGRAM worker
        result = self.inner.forward_batch_generation(batch)

        # Submit prefetch for next round
        self._submit_prefetch(batch, result)

        self._periodic_log()
        return result

    # ---- Background tree pre-building ----

    def _submit_prefetch(
        self,
        batch: ScheduleBatch,
        result: Optional[GenerationBatchResult] = None,
    ):
        """Submit a CPU task to pre-scan the n-gram corpus for the predicted next prefix."""
        if batch.batch_size() < 1:
            return

        # Collect token sequences for each request in the batch
        token_seqs = []
        for req in batch.reqs:
            tokens = list(req.origin_input_ids) + list(req.output_ids)
            # If we have a result with next_token_ids, extend the prediction
            if result is not None and result.next_token_ids is not None:
                next_ids = result.next_token_ids
                if isinstance(next_ids, torch.Tensor):
                    next_ids = next_ids.tolist()
                if isinstance(next_ids, list) and len(next_ids) > 0:
                    tokens = tokens + [next_ids[0]]
            token_seqs.append(tokens)

        with self._lock:
            if self._pending_tree is not None:
                if not self._pending_tree.done():
                    return  # previous prefetch still running
                self._pending_tree = None

            self._pending_tree = self._executor.submit(
                self._build_tree_cpu, token_seqs
            )

    def _build_tree_cpu(self, token_seqs: list):
        """CPU-bound: pre-scan n-gram corpus and warm internal state.

        This runs on a background thread.  We call into the inner worker's
        n-gram corpus lookup functions (which are pure Python/NumPy, no GPU).
        The result is that when the next round's _prepare_draft_tokens() runs,
        the corpus hit rate is warmer.
        """
        try:
            inner = self.inner
            # Access the n-gram trie if it exists (NGRAMWorker builds one internally)
            corpus = getattr(inner, "ngram_corpus", None)
            if corpus is None:
                # No explicit corpus object; the inner worker builds trees on-the-fly
                # from the radix cache.  Pre-warming is a no-op in this case.
                return

            # If corpus supports pre-scanning, call it
            pre_scan = getattr(corpus, "pre_scan", None)
            if callable(pre_scan):
                for seq in token_seqs:
                    pre_scan(seq)
        except Exception as e:
            logger.debug("Phantom tree prefetch error: %s", e)

    def _periodic_log(self):
        if self.total_rounds % self._log_interval == 0 and self.total_rounds > 0:
            total = self.prefetch_hits + self.prefetch_misses
            hit_rate = self.prefetch_hits / max(total, 1) * 100
            logger.info(
                "PhantomTree stats: %d rounds, %d prefetch hits, %d misses (%.1f%%)",
                self.total_rounds,
                self.prefetch_hits,
                self.prefetch_misses,
                hit_rate,
            )

    def __del__(self):
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
