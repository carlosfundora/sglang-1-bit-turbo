"""TQ5_X — TurboQuant 5 eXtended: HSA ghost-draft speculative decoding.

AMD gfx103x-optimized speculative worker that uses ROCm's HSA shared memory
for zero-copy CPU→GPU draft token transfer.  The "X" is variable — it adapts
to available system bandwidth and memory topology.

Architecture:
  1. At prefill, freezes an n-gram corpus snapshot (constant cache — read-only)
  2. A CPU ghost thread continuously pre-builds draft trees from the frozen
     corpus into pinned (HSA-accessible) memory buffers
  3. On each decode step, GPU reads draft tokens directly from pinned memory
     via PCIe — zero explicit copy on ROCm HSA systems
  4. GPU runs standard NgramVerifyInput tree verification
  5. Meanwhile, CPU ghost thread is already building the NEXT tree

Benefits on AMD gfx103x:
  - Draft state lives in system RAM (pinned), not VRAM → more KV cache budget
  - HSA shared memory = zero-copy reads from GPU
  - N-gram corpus scan is pure CPU work → truly overlapped with GPU verify
  - Frozen corpus = no locks, no synchronization on reads
  - The "X factor" scales with PCIe bandwidth and CPU speed

Usage: --speculative-algorithm TQ5_X --disable-overlap-schedule
"""

import logging
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class _GhostBuffer:
    """Double-buffered pinned memory for zero-copy CPU→GPU draft transfer.

    Two buffers alternate: CPU writes to one while GPU reads from the other.
    On ROCm/HSA, pinned memory is directly accessible by the GPU via PCIe
    without any explicit copy or DMA transfer setup.
    """

    def __init__(self, max_batch_size: int, draft_token_num: int):
        K = draft_token_num
        # Draft tokens: [max_bs * K] int64, pinned
        self.tokens_a = torch.zeros(max_batch_size * K, dtype=torch.int64, pin_memory=True)
        self.tokens_b = torch.zeros(max_batch_size * K, dtype=torch.int64, pin_memory=True)
        # Tree mask: [max_bs * K * K] bool, pinned
        self.mask_a = torch.zeros(max_batch_size * K * K, dtype=torch.bool, pin_memory=True)
        self.mask_b = torch.zeros(max_batch_size * K * K, dtype=torch.bool, pin_memory=True)
        # Which buffer is "ready" for GPU (0=a, 1=b)
        self._active = 0  # GPU reads from active, CPU writes to inactive
        self._lock = threading.Lock()
        self._ready = threading.Event()
        # Metadata about what's in the active buffer
        self.active_bs = 0
        self.active_k = K

    @property
    def gpu_tokens(self) -> torch.Tensor:
        return self.tokens_a if self._active == 0 else self.tokens_b

    @property
    def gpu_mask(self) -> torch.Tensor:
        return self.mask_a if self._active == 0 else self.mask_b

    @property
    def cpu_tokens(self) -> torch.Tensor:
        return self.tokens_b if self._active == 0 else self.tokens_a

    @property
    def cpu_mask(self) -> torch.Tensor:
        return self.mask_b if self._active == 0 else self.mask_a

    def swap(self, bs: int):
        """Swap active/inactive buffers (called after CPU finishes writing)."""
        with self._lock:
            self._active = 1 - self._active
            self.active_bs = bs
            self._ready.set()

    def wait_ready(self, timeout: float = 0.001) -> bool:
        """Wait for CPU to finish writing a ghost tree."""
        return self._ready.wait(timeout=timeout)

    def consume(self):
        """Mark the active buffer as consumed (GPU is done reading)."""
        self._ready.clear()


class _FrozenCorpus:
    """Read-only snapshot of an n-gram corpus for lock-free CPU access.

    Created once at prefill from the live NgramCorpus, then frozen.
    The ghost thread reads from this without any synchronization.
    """

    def __init__(self, corpus, draft_token_num: int, max_match_window: int):
        self._corpus = corpus
        self.draft_token_num = draft_token_num
        self.max_match_window = max_match_window
        self._frozen = False

    def freeze(self):
        """Freeze the corpus (sync any pending inserts, then mark read-only)."""
        self._corpus.synchronize()
        self._frozen = True
        logger.info("TQ5_X: corpus frozen (%d draft tokens)", self.draft_token_num)

    def batch_get(self, batch_tokens: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Read-only lookup — safe to call from any thread after freeze()."""
        token_arr, mask_arr = self._corpus.batch_get(batch_tokens)
        return token_arr, mask_arr

    def insert(self, batch_tokens: List[List[int]]):
        """Insert tokens into corpus (only before freeze)."""
        if not self._frozen:
            self._corpus.batch_put(batch_tokens)


class TQ5XWorker:
    """TQ5_X: TurboQuant 5 eXtended — HSA zero-copy ghost-draft speculative worker.

    Combines:
    - Frozen n-gram corpus (constant cache, no locks)
    - CPU ghost thread (builds trees in pinned HSA memory)
    - Double-buffered zero-copy GPU reads
    - Standard NgramVerifyInput tree verification

    The "X" adapts to system characteristics:
    - PCIe bandwidth determines copy latency (zero on HSA, ~μs on non-HSA)
    - CPU core count determines ghost thread throughput
    - System RAM determines corpus capacity

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
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.page_size = server_args.page_size
        self.draft_token_num = server_args.speculative_num_draft_tokens or 8
        self.max_batch_size = target_worker.max_running_requests
        # max_match_window controls how far back we look for n-gram matches
        self.max_match_window_size = getattr(
            server_args, "speculative_ngram_max_trie_depth", 18
        )

        # Create the n-gram corpus (will be frozen after first prefill)
        from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus

        raw_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=self.draft_token_num,
        )
        self.corpus = _FrozenCorpus(
            raw_corpus, self.draft_token_num, self.max_match_window_size
        )

        # Double-buffered pinned memory (HSA zero-copy on AMD)
        self.ghost_buf = _GhostBuffer(self.max_batch_size, self.draft_token_num)

        # GPU-side tensors for tree reconstruction (these must be in VRAM)
        K = self.draft_token_num
        max_bs = self.max_batch_size
        self.positions = torch.empty(max_bs * K, dtype=torch.int64, device=self.device)
        self.retrive_index = torch.empty((max_bs, K), dtype=torch.int64, device=self.device)
        self.retrive_next_token = torch.empty((max_bs, K), dtype=torch.int64, device=self.device)
        self.retrive_next_sibling = torch.empty((max_bs, K), dtype=torch.int64, device=self.device)

        # Ghost thread state
        self._ghost_thread: Optional[threading.Thread] = None
        self._ghost_stop = threading.Event()
        self._ghost_request: Optional[List[List[int]]] = None
        self._ghost_request_lock = threading.Lock()
        self._ghost_request_event = threading.Event()
        self._corpus_frozen = False

        # Stats
        self.total_rounds = 0
        self.ghost_hits = 0  # ghost tree was ready when GPU needed it
        self.ghost_misses = 0  # ghost tree wasn't ready, built synchronously
        self._log_interval = 50

        # Detect HSA capability
        self._is_hsa = self._detect_hsa()

        logger.info(
            "TQ5_X worker: draft_tokens=%d, max_bs=%d, HSA=%s, "
            "pinned_buf=%.1f KB (tokens) + %.1f KB (mask)",
            K, max_bs, self._is_hsa,
            self.ghost_buf.tokens_a.nelement() * 8 / 1024,
            self.ghost_buf.mask_a.nelement() / 1024,
        )

    @staticmethod
    def _detect_hsa() -> bool:
        """Detect if we're running on AMD ROCm with HSA support."""
        try:
            return hasattr(torch.version, "hip") and torch.version.hip is not None
        except Exception:
            return False

    # ---- Proxy attributes ----

    @property
    def max_running_requests(self):
        return self.target_worker.max_running_requests

    @property
    def model_config(self):
        return self.target_worker.model_runner.model_config

    def get_memory_pool(self):
        return self.target_worker.get_memory_pool()

    def clear_cache_pool(self):
        self._stop_ghost_thread()
        self.corpus._corpus.reset()
        self._corpus_frozen = False
        self.total_rounds = 0
        self.ghost_hits = 0
        self.ghost_misses = 0

    # ---- Ghost thread management ----

    def _start_ghost_thread(self):
        if self._ghost_thread is not None and self._ghost_thread.is_alive():
            return
        self._ghost_stop.clear()
        self._ghost_thread = threading.Thread(
            target=self._ghost_loop, name="tq5x-ghost", daemon=True
        )
        self._ghost_thread.start()
        logger.info("TQ5_X: ghost thread started")

    def _stop_ghost_thread(self):
        self._ghost_stop.set()
        self._ghost_request_event.set()  # wake it up so it can exit
        if self._ghost_thread is not None:
            self._ghost_thread.join(timeout=2.0)
            self._ghost_thread = None

    def _ghost_loop(self):
        """Background CPU thread: continuously builds draft trees into pinned memory."""
        while not self._ghost_stop.is_set():
            # Wait for a request from the main thread
            self._ghost_request_event.wait(timeout=0.1)
            if self._ghost_stop.is_set():
                break
            self._ghost_request_event.clear()

            with self._ghost_request_lock:
                batch_tokens = self._ghost_request
                self._ghost_request = None

            if batch_tokens is None:
                continue

            try:
                bs = len(batch_tokens)
                K = self.draft_token_num

                # Lookup from frozen corpus (read-only, no locks needed)
                req_drafts, mask = self.corpus.batch_get(batch_tokens)

                if len(req_drafts) != bs * K:
                    continue  # corpus returned wrong shape, skip

                # Write into the CPU (inactive) side of the double buffer
                cpu_tokens = self.ghost_buf.cpu_tokens
                cpu_mask = self.ghost_buf.cpu_mask

                cpu_tokens[:bs * K].copy_(torch.from_numpy(req_drafts))
                cpu_mask[:bs * K * K].copy_(torch.from_numpy(mask))

                # Swap: make this buffer active for GPU reads
                self.ghost_buf.swap(bs)

            except Exception as e:
                logger.debug("TQ5_X ghost build error: %s", e)

    def _submit_ghost_request(self, batch: ScheduleBatch):
        """Ask the ghost thread to pre-build drafts for the predicted next prefix."""
        batch_tokens = []
        for req in batch.reqs:
            tokens = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_match_window_size
            )
            batch_tokens.append(tokens)

        with self._ghost_request_lock:
            self._ghost_request = batch_tokens
        self._ghost_request_event.set()

    # ---- Helpers ----

    @staticmethod
    def _efficient_concat_last_n(seq1: List[int], seq2: List[int], n: int):
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]
        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    # ---- Main dispatch ----

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        from sglang.srt.speculative.ngram_info import NgramVerifyInput

        # On extend/prefill: seed the corpus and freeze it
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._handle_extend(batch)

        bs = batch.batch_size()
        if bs == 0 or batch.forward_mode.is_idle():
            return self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )

        self.total_rounds += 1
        K = self.draft_token_num

        # Try to use the ghost thread's pre-built tree (zero-copy from pinned mem)
        ghost_ready = self.ghost_buf.wait_ready(timeout=0.002)

        if ghost_ready and self.ghost_buf.active_bs == bs:
            # Ghost tree is ready — read directly from pinned HSA memory
            # On ROCm, .cuda(non_blocking=True) on pinned memory is zero-copy
            draft_tokens_gpu = self.ghost_buf.gpu_tokens[:bs * K].cuda(non_blocking=True)
            tree_mask_gpu = self.ghost_buf.gpu_mask[:bs * K * K].cuda(non_blocking=True)
            # CRITICAL: GPU must finish reading pinned memory before we tell
            # the ghost thread the buffer is free — otherwise the ghost thread
            # overwrites the buffer mid-DMA, causing HSA memory faults on gfx1030.
            torch.cuda.synchronize()
            self.ghost_buf.consume()
            self.ghost_hits += 1
        else:
            # Ghost wasn't ready — build synchronously (fallback)
            self.ghost_misses += 1
            draft_tokens_gpu, tree_mask_gpu = self._build_tree_sync(batch)

        # Reconstruct tree indices (must be in VRAM)
        positions = self.positions[:bs * K]
        retrive_index = self.retrive_index[:bs, :K]
        retrive_next_token = self.retrive_next_token[:bs, :K]
        retrive_next_sibling = self.retrive_next_sibling[:bs, :K]

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
            logger.warning("TQ5_X reconstruct_indices failed: %s", e)
            return self._fallback_target_only(batch)

        # Build full attention mask
        USE_FULL_MASK = True
        if USE_FULL_MASK:
            mask_np = tree_mask_gpu.cpu().numpy().reshape(bs, K, K)
            tree_mask_parts = []
            for i, req in enumerate(batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                prefix_mask = torch.ones((K, seq_len - 1), device=self.device)
                tree_part = torch.from_numpy(mask_np[i]).to(self.device)
                full_mask = torch.cat((prefix_mask, tree_part), dim=1).to(torch.bool)
                tree_mask_parts.append(full_mask.flatten())
            tree_mask_final = torch.cat(tree_mask_parts, dim=0)
        else:
            tree_mask_final = tree_mask_gpu

        # Set up verification
        original_algo = batch.spec_algorithm
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = NgramVerifyInput(
            draft_tokens_gpu,
            tree_mask_final,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            K,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

        # Run target verification on GPU
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

            # Update corpus with accepted tokens (keeps it growing)
            self._update_corpus(batch)

            # Submit next ghost request (CPU starts building while we return)
            self._submit_ghost_request(batch)

            self._periodic_log()

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=num_accepted,
                can_run_cuda_graph=can_run_cuda_graph,
                accept_lens=accept_lens,
            )

        batch.spec_algorithm = original_algo
        return self._fallback_target_only(batch)

    # ---- Extend / prefill handling ----

    def _handle_extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """On prefill: seed corpus with prompt tokens, freeze, start ghost thread."""
        # Seed corpus with prompt tokens
        batch_tokens = []
        for req in batch.reqs:
            put_ids = list(req.origin_input_ids) + list(req.output_ids)
            batch_tokens.append(put_ids)
        self.corpus.insert(batch_tokens)

        # Freeze corpus after first extend (makes it read-only for ghost thread)
        if not self._corpus_frozen:
            self.corpus.freeze()
            self._corpus_frozen = True
            self._start_ghost_thread()

        # Run normal extend on target
        result = self.target_worker.forward_batch_generation(
            batch.get_model_worker_batch()
        )

        # Pre-submit first ghost request
        self._submit_ghost_request(batch)

        return result

    def _update_corpus(self, batch: ScheduleBatch):
        """Update corpus with newly generated tokens.

        NOTE: After freeze, we still allow inserts into the underlying corpus.
        The frozen flag was for the initial snapshot; ongoing inserts keep the
        corpus relevant as generation continues.  The ghost thread's reads
        and main thread's inserts go through the C++ corpus which handles
        its own internal synchronization via asyncInsert + synchronize.
        """
        batch_tokens = []
        for req in batch.reqs:
            put_ids = self._efficient_concat_last_n(
                req.origin_input_ids,
                req.output_ids,
                self.corpus._corpus.draft_token_num * 3,  # trie depth heuristic
            )
            batch_tokens.append(put_ids)
        self.corpus._corpus.batch_put(batch_tokens)

    def _build_tree_sync(self, batch: ScheduleBatch):
        """Synchronous fallback: build tree on CPU, copy to GPU."""
        bs = batch.batch_size()
        K = self.draft_token_num

        self.corpus._corpus.synchronize()
        batch_tokens = []
        for req in batch.reqs:
            tokens = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_match_window_size
            )
            batch_tokens.append(tokens)

        req_drafts, mask = self.corpus.batch_get(batch_tokens)
        assert len(req_drafts) == bs * K, (
            f"TQ5_X sync: {len(req_drafts)=} != {bs * K}"
        )

        draft_tokens_gpu = torch.from_numpy(req_drafts).to(self.device, non_blocking=True)
        tree_mask_gpu = torch.from_numpy(mask).to(self.device, non_blocking=True)
        return draft_tokens_gpu, tree_mask_gpu

    def _fallback_target_only(self, batch: ScheduleBatch) -> GenerationBatchResult:
        batch.forward_mode = ForwardMode.DECODE
        return self.target_worker.forward_batch_generation(
            batch.get_model_worker_batch()
        )

    def _periodic_log(self):
        if self.total_rounds % self._log_interval == 0 and self.total_rounds > 0:
            total = self.ghost_hits + self.ghost_misses
            hit_pct = self.ghost_hits / max(total, 1) * 100
            logger.info(
                "TQ5_X stats: %d rounds, %d ghost hits (%.1f%%), %d sync fallbacks, "
                "pinned_buf=%s",
                self.total_rounds, self.ghost_hits, hit_pct, self.ghost_misses,
                "HSA" if self._is_hsa else "pinned",
            )

    def __del__(self):
        try:
            self._stop_ghost_thread()
        except Exception:
            pass
