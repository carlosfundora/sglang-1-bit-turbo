"""PHANTOM — HSA zero-copy ghost-draft speculative decoding.

AMD gfx103x-optimized speculative worker that uses ROCm's HSA shared memory
for zero-copy CPU→GPU draft token transfer.

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

Usage: --speculative-algorithm PHANTOM --disable-overlap-schedule
"""

import logging
import os
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

try:
    from sgl_kernel.speculative import reconstruct_indices_from_tree_mask
except ImportError:
    reconstruct_indices_from_tree_mask = None

logger = logging.getLogger(__name__)


class _GhostBuffer:
    """N-buffered pinned memory for zero-copy CPU→GPU draft transfer.

    Ring buffer of N slots: CPU writes to the next free slot while GPU reads
    from the oldest ready slot. On ROCm/HSA, pinned memory is directly
    accessible by the GPU via PCIe without explicit copy.

    N=1: sync-only (debug mode, no CPU/GPU overlap)
    N=2: classic double-buffering (default, matches original behavior)
    N=3: one extra lookahead buffer for slow ghost threads
    N=4: maximum decoupling
    """

    def __init__(self, max_batch_size: int, draft_token_num: int, num_buffers: int = 2):
        assert 1 <= num_buffers <= 4, f"num_buffers must be 1-4, got {num_buffers}"
        K = draft_token_num
        self.num_buffers = num_buffers

        # Allocate N pinned buffer pairs
        self._tokens = [
            torch.zeros(max_batch_size * K, dtype=torch.int64, pin_memory=True)
            for _ in range(num_buffers)
        ]
        self._masks = [
            torch.zeros(max_batch_size * K * K, dtype=torch.bool, pin_memory=True)
            for _ in range(num_buffers)
        ]

        # Ring buffer indices
        self._write_idx = 0  # next slot for CPU to write
        self._read_idx = 0   # next slot for GPU to read
        self._ready_count = 0  # how many slots have valid data
        self._lock = threading.Lock()
        self._ready = threading.Event()

        # Metadata per slot
        self._slot_bs = [0] * num_buffers
        self.active_bs = 0
        self.active_k = K

    # ── Backward-compatible properties (for 2-buffer callers) ──

    @property
    def gpu_tokens(self) -> torch.Tensor:
        return self._tokens[self._read_idx]

    @property
    def gpu_mask(self) -> torch.Tensor:
        return self._masks[self._read_idx]

    @property
    def cpu_tokens(self) -> torch.Tensor:
        return self._tokens[self._write_idx]

    @property
    def cpu_mask(self) -> torch.Tensor:
        return self._masks[self._write_idx]

    def swap(self, bs: int):
        """CPU finished writing — mark slot ready, advance write pointer."""
        with self._lock:
            self._slot_bs[self._write_idx] = bs
            self._write_idx = (self._write_idx + 1) % self.num_buffers
            self._ready_count = min(self._ready_count + 1, self.num_buffers - 1)
            self.active_bs = bs
            self._ready.set()

    def wait_ready(self, timeout: float = 0.001) -> bool:
        """Wait for at least one ready slot."""
        return self._ready.wait(timeout=timeout)

    def consume(self):
        """GPU done reading — release slot, advance read pointer."""
        with self._lock:
            self._read_idx = (self._read_idx + 1) % self.num_buffers
            self._ready_count = max(self._ready_count - 1, 0)
            if self._ready_count == 0:
                self._ready.clear()
            # Update active_bs to next ready slot if available
            if self._ready_count > 0:
                self.active_bs = self._slot_bs[self._read_idx]


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
        logger.info("PHANTOM: corpus frozen (%d draft tokens)", self.draft_token_num)

    def batch_get(self, batch_tokens: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Read-only lookup — safe to call from any thread after freeze()."""
        token_arr, mask_arr = self._corpus.batch_get(batch_tokens)
        return token_arr, mask_arr

    def insert(self, batch_tokens: List[List[int]]):
        """Insert tokens into corpus (only before freeze)."""
        if not self._frozen:
            self._corpus.batch_put(batch_tokens)


class PhantomWorker:
    """PHANTOM — HSA zero-copy ghost-draft speculative worker.

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

        # N-buffered pinned memory (HSA zero-copy on AMD)
        num_buffers = getattr(server_args, 'phantom_num_buffers', 2)
        self.ghost_buf = _GhostBuffer(self.max_batch_size, self.draft_token_num, num_buffers)

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

        # Adaptive metrics (sliding window)
        self._accept_window = deque(maxlen=20)  # recent acceptance rates
        self._corpus_hit_count = 0  # non-trivial drafts from corpus
        self._corpus_miss_count = 0  # all-zero / failed drafts
        self._corpus_insert_count = 0  # total corpus inserts

        # Auto-fallback state
        self._fallback_active = False
        self._fallback_streak = 0  # consecutive low-acceptance rounds
        self._fallback_probe_counter = 0
        _FALLBACK_THRESHOLD = 0.4
        _FALLBACK_STREAK_LIMIT = 10
        _REENABLE_THRESHOLD = 0.5
        _PROBE_INTERVAL = 5
        self._fb_threshold = _FALLBACK_THRESHOLD
        self._fb_streak_limit = _FALLBACK_STREAK_LIMIT
        self._fb_reenable = _REENABLE_THRESHOLD
        self._fb_probe_interval = _PROBE_INTERVAL

        # Dynamic γ state
        self._initial_draft_num = self.draft_token_num
        self._min_draft = 2
        self._max_draft = min(self._initial_draft_num * 2, 16)

        # Detect HSA capability
        self._is_hsa = self._detect_hsa()

        logger.info(
            "PHANTOM worker: draft_tokens=%d, max_bs=%d, HSA=%s, "
            "buffers=%d, pinned_buf=%.1f KB (tokens) + %.1f KB (mask) per slot",
            K, max_bs, self._is_hsa, self.ghost_buf.num_buffers,
            self.ghost_buf._tokens[0].nelement() * 8 / 1024,
            self.ghost_buf._masks[0].nelement() / 1024,
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
        self._accept_window.clear()
        self._corpus_hit_count = 0
        self._corpus_miss_count = 0
        self._corpus_insert_count = 0
        self._fallback_active = False
        self._fallback_streak = 0
        self._fallback_probe_counter = 0
        self.draft_token_num = self._initial_draft_num

    # ---- Ghost thread management ----

    def _start_ghost_thread(self):
        if self._ghost_thread is not None and self._ghost_thread.is_alive():
            return
        self._ghost_stop.clear()
        self._ghost_thread = threading.Thread(
            target=self._ghost_loop, name="phantom-ghost", daemon=True
        )
        self._ghost_thread.start()

        # Pin ghost thread to last available CPU core to reduce contention
        try:
            available = sorted(os.sched_getaffinity(0))
            if len(available) > 1:
                target_core = available[-1]
                # Must set from within the thread or use its native id
                self._ghost_affinity_core = target_core
                os.sched_setaffinity(self._ghost_thread.native_id, {target_core})
                logger.info("PHANTOM: ghost thread pinned to core %d", target_core)
            else:
                logger.info("PHANTOM: ghost thread started (single core, no affinity)")
        except (OSError, AttributeError):
            logger.info("PHANTOM: ghost thread started (affinity not available)")

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
                logger.debug("PHANTOM ghost build error: %s", e)

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

        # ── Auto-fallback: skip spec decode when acceptance is consistently low ──
        if self._fallback_active:
            self._fallback_probe_counter += 1
            if self._fallback_probe_counter < self._fb_probe_interval:
                return self._fallback_target_only(batch)
            # Probe round: try spec decode, check if acceptance recovered
            self._fallback_probe_counter = 0

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
            if reconstruct_indices_from_tree_mask is None:
                raise ImportError("sgl_kernel.speculative not available")

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
            logger.warning("PHANTOM reconstruct_indices failed: %s", e)
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

            # ── Track acceptance rate ──
            accept_rate = num_accepted / max(bs * K, 1)
            self._accept_window.append(accept_rate)

            # ── Auto-fallback logic ──
            if accept_rate < self._fb_threshold:
                self._fallback_streak += 1
            else:
                self._fallback_streak = 0

            if self._fallback_streak >= self._fb_streak_limit and not self._fallback_active:
                self._fallback_active = True
                self._fallback_probe_counter = 0
                logger.info("PHANTOM: auto-fallback ENABLED (acceptance=%.2f for %d rounds)",
                            accept_rate, self._fallback_streak)
            elif self._fallback_active and accept_rate >= self._fb_reenable:
                self._fallback_active = False
                self._fallback_streak = 0
                logger.info("PHANTOM: auto-fallback DISABLED (acceptance recovered to %.2f)",
                            accept_rate)

            # ── Dynamic γ: adjust draft length based on acceptance ──
            if len(self._accept_window) >= 5:
                avg_accept = sum(self._accept_window) / len(self._accept_window)
                if avg_accept > 0.7 and self.draft_token_num < self._max_draft:
                    self.draft_token_num = min(self.draft_token_num + 1, self._max_draft)
                    logger.debug("PHANTOM: γ increased to %d (avg_accept=%.2f)",
                                 self.draft_token_num, avg_accept)
                elif avg_accept < 0.3 and self.draft_token_num > self._min_draft:
                    self.draft_token_num = max(self.draft_token_num - 1, self._min_draft)
                    logger.debug("PHANTOM: γ decreased to %d (avg_accept=%.2f)",
                                 self.draft_token_num, avg_accept)

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

        Token window is bounded to `max_match_window_size` to limit corpus
        memory growth. Only the most recent tokens per request are inserted.
        """
        max_window = min(
            self.max_match_window_size,
            self.corpus._corpus.draft_token_num * 3,
        )
        batch_tokens = []
        for req in batch.reqs:
            put_ids = self._efficient_concat_last_n(
                req.origin_input_ids,
                req.output_ids,
                max_window,
            )
            batch_tokens.append(put_ids)
        self.corpus._corpus.batch_put(batch_tokens)
        self._corpus_insert_count += 1

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
        # Track corpus hit rate (non-trivial = at least one non-zero draft token)
        if np.any(req_drafts != 0):
            self._corpus_hit_count += 1
        else:
            self._corpus_miss_count += 1
        assert len(req_drafts) == bs * K, (
            f"PHANTOM sync: {len(req_drafts)=} != {bs * K}"
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
            ghost_pct = self.ghost_hits / max(total, 1) * 100
            avg_accept = (sum(self._accept_window) / len(self._accept_window)
                          if self._accept_window else 0.0)
            corpus_total = self._corpus_hit_count + self._corpus_miss_count
            corpus_pct = self._corpus_hit_count / max(corpus_total, 1) * 100
            logger.info(
                "PHANTOM stats: rounds=%d, ghost=%.1f%% (%d/%d), "
                "accept=%.2f, γ=%d, corpus_hit=%.1f%%, fallback=%s, buf=%s",
                self.total_rounds, ghost_pct, self.ghost_hits, total,
                avg_accept, self.draft_token_num, corpus_pct,
                "ON" if self._fallback_active else "off",
                "HSA" if self._is_hsa else "pinned",
            )

    def __del__(self):
        try:
            self._stop_ghost_thread()
        except Exception:
            pass
