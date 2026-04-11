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
from sglang.srt.speculative.draft_prefilter import AdaptiveThresholdController
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


class _NegativeFilter:
    """Bloom filter tracking historically-rejected n-gram draft patterns.

    After each verify step, rejected draft token bigrams are inserted.
    The ghost thread scores candidate sequences against the filter to
    prefer drafts with fewer known-bad patterns.

    Uses a compact bytearray-based bloom filter with 2 hash functions.
    At 64K bits (8KB) and ~1000 patterns, false positive rate is ~1%.

    Aging: resets every `age_limit` insertions to prevent saturation.
    """

    def __init__(self, num_bits: int = 65536, age_limit: int = 2000):
        self._bits = bytearray(num_bits // 8)
        self._num_bits = num_bits
        self._count = 0
        self._age_limit = age_limit

    def _hash1(self, val: int) -> int:
        h = (val ^ 0x811C9DC5) * 0x01000193
        return h % self._num_bits

    def _hash2(self, val: int) -> int:
        h = ((val >> 16) ^ val) * 0x45D9F3B
        h = ((h >> 16) ^ h) * 0x45D9F3B
        return ((h >> 16) ^ h) % self._num_bits

    def insert_bigram(self, tok_a: int, tok_b: int):
        """Insert a rejected bigram pattern. Auto-ages when saturated."""
        if self._count >= self._age_limit:
            self._bits = bytearray(self._num_bits // 8)
            self._count = 0
            logger.debug("PHANTOM: negative filter aged (reset after %d)", self._age_limit)
        val = tok_a * 131071 + tok_b
        idx1 = self._hash1(val)
        idx2 = self._hash2(val)
        self._bits[idx1 // 8] |= (1 << (idx1 % 8))
        self._bits[idx2 // 8] |= (1 << (idx2 % 8))
        self._count += 1

    def query_bigram(self, tok_a: int, tok_b: int) -> bool:
        """Check if a bigram was previously rejected (may have false positives)."""
        val = tok_a * 131071 + tok_b
        idx1 = self._hash1(val)
        idx2 = self._hash2(val)
        return (
            (self._bits[idx1 // 8] & (1 << (idx1 % 8))) != 0
            and (self._bits[idx2 // 8] & (1 << (idx2 % 8))) != 0
        )

    def score_sequence(self, tokens: np.ndarray) -> float:
        """Score a draft sequence: fraction of bigrams NOT in the filter (higher=better)."""
        if len(tokens) < 2:
            return 1.0
        good = 0
        total = len(tokens) - 1
        for i in range(total):
            if not self.query_bigram(int(tokens[i]), int(tokens[i + 1])):
                good += 1
        return good / total

    def reset(self):
        """Clear the filter."""
        self._bits = bytearray(self._num_bits // 8)
        self._count = 0

    @property
    def size(self) -> int:
        return self._count


class _QuantTelemetry:
    """Lightweight inference telemetry for quantization-aware calibration.

    Accumulates signals during PHANTOM inference that inform which parts
    of the model are producing bad logits and can be quantized aggressively.

    All signals are collected from data already computed during the verify
    step — zero additional GPU forward passes required.

    Collected signals:
      1. Channel importance (running mean of |hidden_states| per dim)
         → hot channels need precision, cold channels can be crushed
      2. Logit margin at accept/reject boundaries
         → small margin = quant noise would flip the decision
      3. Token confusion pairs (draft vs actual when rejected)
         → embedding rows that confuse each other need precision
      4. Position-wise acceptance rate
         → which tree depths the model struggles with

    Periodic snapshots are written to disk for offline analysis by
    a quantization calibration tool.
    """

    def __init__(self, hidden_dim: int = 0, max_confusion_pairs: int = 8192,
                 snapshot_interval: int = 200, snapshot_dir: Optional[str] = None):
        self.hidden_dim = hidden_dim
        self._snapshot_interval = snapshot_interval
        self._snapshot_dir = snapshot_dir
        self._step = 0

        # Signal 1: channel importance — EMA of |activation| per hidden dim
        # Initialized lazily on first call (hidden_dim may not be known yet)
        self._channel_sum: Optional[np.ndarray] = None
        self._channel_count: int = 0

        # Signal 2: logit margin at decision boundaries
        # margin = logit[top1] - logit[top2] for accepted/rejected positions
        self._margin_accepted = deque(maxlen=5000)   # margins where draft was right
        self._margin_rejected = deque(maxlen=5000)   # margins where draft was wrong

        # Signal 3: confusion pairs — (draft_tok, actual_tok) frequency
        self._confusion: dict = {}  # {(draft, actual): count}
        self._max_confusion = max_confusion_pairs

        # Signal 4: position-wise acceptance histogram
        self._pos_accepted = np.zeros(32, dtype=np.int64)  # up to 32 draft positions
        self._pos_total = np.zeros(32, dtype=np.int64)

    def record(self, logits: torch.Tensor, hidden_states: Optional[torch.Tensor],
               draft_tokens: torch.Tensor, verified_ids: torch.Tensor,
               accept_lens: torch.Tensor, bs: int, K: int):
        """Record one round of telemetry from verify outputs.

        Args:
            logits: [num_tokens, vocab_size] — final logits from verify pass
            hidden_states: [num_tokens, hidden_dim] or None — last-layer activations
            draft_tokens: [bs * K] — what PHANTOM proposed
            verified_ids: [bs] — what the model actually picked
            accept_lens: [bs] — how many draft tokens were accepted per request
            bs: batch size
            K: draft length
        """
        self._step += 1

        try:
            self._record_channel_importance(hidden_states)
            self._record_logit_margins(logits, accept_lens, bs, K)
            self._record_confusion(draft_tokens, verified_ids, accept_lens, bs, K)
            self._record_position_acceptance(accept_lens, K)
        except Exception:
            pass  # telemetry is best-effort

        if (self._snapshot_dir and self._snapshot_interval > 0
                and self._step % self._snapshot_interval == 0):
            self._write_snapshot()

    def _record_channel_importance(self, hidden_states: Optional[torch.Tensor]):
        """Signal 1: accumulate |activation| per hidden dimension."""
        if hidden_states is None:
            return
        # Move to CPU, take abs mean across token dimension
        h = hidden_states.float().abs().mean(dim=0).cpu().numpy()
        if self._channel_sum is None:
            self._channel_sum = np.zeros_like(h)
            self.hidden_dim = len(h)
        self._channel_sum += h
        self._channel_count += 1

    def _record_logit_margins(self, logits: torch.Tensor,
                              accept_lens: torch.Tensor, bs: int, K: int):
        """Signal 2: top1-top2 logit gap at accept/reject boundaries."""
        if logits is None or logits.dim() != 2:
            return
        # Only sample a few positions per round to keep cost near-zero
        top2 = logits.topk(2, dim=-1).values  # [num_tokens, 2]
        margins = (top2[:, 0] - top2[:, 1]).cpu().numpy()

        acc_cpu = accept_lens.cpu().numpy() if accept_lens.is_cuda else accept_lens.numpy()
        idx = 0
        for r in range(bs):
            acc = int(acc_cpu[r])
            for pos in range(min(K, len(margins) - idx)):
                m = float(margins[idx + pos])
                if pos < acc:
                    self._margin_accepted.append(m)
                else:
                    self._margin_rejected.append(m)
            idx += K

    def _record_confusion(self, draft_tokens: torch.Tensor,
                          verified_ids: torch.Tensor,
                          accept_lens: torch.Tensor, bs: int, K: int):
        """Signal 3: which draft tokens the model overrides (confusion pairs)."""
        draft_cpu = draft_tokens.cpu()
        verified_cpu = verified_ids.cpu()
        acc_cpu = accept_lens.cpu()

        for r in range(bs):
            acc = int(acc_cpu[r])
            # The token at position acc is the first rejected draft token;
            # verified_ids[r] is what the model chose instead
            reject_pos = r * K + acc
            if reject_pos < len(draft_cpu) and acc < K:
                draft_tok = int(draft_cpu[reject_pos])
                actual_tok = int(verified_cpu[r])
                if draft_tok != 0 and actual_tok != 0 and draft_tok != actual_tok:
                    pair = (draft_tok, actual_tok)
                    self._confusion[pair] = self._confusion.get(pair, 0) + 1
                    # Prune if too large — keep top-frequency pairs
                    if len(self._confusion) > self._max_confusion:
                        threshold = sorted(self._confusion.values())[-self._max_confusion // 2]
                        self._confusion = {
                            k: v for k, v in self._confusion.items() if v >= threshold
                        }

    def _record_position_acceptance(self, accept_lens: torch.Tensor, K: int):
        """Signal 4: which draft positions succeed/fail."""
        acc_cpu = accept_lens.cpu().numpy() if accept_lens.is_cuda else accept_lens.numpy()
        for acc in acc_cpu:
            acc = int(acc)
            for pos in range(min(K, len(self._pos_total))):
                self._pos_total[pos] += 1
                if pos < acc:
                    self._pos_accepted[pos] += 1

    def _write_snapshot(self):
        """Periodic dump to disk for offline quant calibration."""
        try:
            os.makedirs(self._snapshot_dir, exist_ok=True)
            path = os.path.join(self._snapshot_dir, f"quant_telem_{self._step}.npz")

            data = {"step": self._step}

            if self._channel_sum is not None and self._channel_count > 0:
                data["channel_importance"] = self._channel_sum / self._channel_count

            if self._margin_accepted:
                data["margin_accepted"] = np.array(list(self._margin_accepted))
            if self._margin_rejected:
                data["margin_rejected"] = np.array(list(self._margin_rejected))

            if self._confusion:
                pairs = sorted(self._confusion.items(), key=lambda x: -x[1])[:1000]
                data["confusion_pairs"] = np.array(
                    [(d, a, c) for (d, a), c in pairs], dtype=np.int64
                )

            data["pos_accepted"] = self._pos_accepted.copy()
            data["pos_total"] = self._pos_total.copy()

            np.savez_compressed(path, **data)
            logger.info("PHANTOM quant telemetry snapshot: %s (%d steps)", path, self._step)
        except Exception as e:
            logger.debug("PHANTOM quant telemetry write failed: %s", e)

    def get_summary(self) -> dict:
        """Summary for periodic log."""
        avg_margin_acc = (
            sum(self._margin_accepted) / len(self._margin_accepted)
            if self._margin_accepted else 0.0
        )
        avg_margin_rej = (
            sum(self._margin_rejected) / len(self._margin_rejected)
            if self._margin_rejected else 0.0
        )
        return {
            "steps": self._step,
            "channel_samples": self._channel_count,
            "margin_acc": round(avg_margin_acc, 3),
            "margin_rej": round(avg_margin_rej, 3),
            "confusion_pairs": len(self._confusion),
            "pos_rate": [
                round(float(self._pos_accepted[i]) / max(float(self._pos_total[i]), 1), 2)
                for i in range(min(8, len(self._pos_total)))
                if self._pos_total[i] > 0
            ],
        }

    def reset(self):
        """Clear all accumulated telemetry."""
        self._step = 0
        self._channel_sum = None
        self._channel_count = 0
        self._margin_accepted.clear()
        self._margin_rejected.clear()
        self._confusion.clear()
        self._pos_accepted[:] = 0
        self._pos_total[:] = 0


class _GhostJob:
    """Immutable job descriptor for one ghost-thread round.

    Freezes batch_tokens, K, bs, and a monotonic job_id so the ghost thread
    and consumer can detect stale results even when batch size stays the same.
    """
    __slots__ = ("job_id", "batch_tokens", "K", "bs", "num_variants")

    def __init__(self, job_id: int, batch_tokens: List[List[int]],
                 K: int, bs: int, num_variants: int = 1):
        self.job_id = job_id
        self.batch_tokens = batch_tokens
        self.K = K
        self.bs = bs
        self.num_variants = num_variants


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

        # Ghost thread state — uses immutable _GhostJob for thread safety
        self._ghost_thread: Optional[threading.Thread] = None
        self._ghost_stop = threading.Event()
        self._ghost_job: Optional[_GhostJob] = None
        self._ghost_job_lock = threading.Lock()
        self._ghost_request_event = threading.Event()
        self._corpus_frozen = False
        self._job_counter = 0  # monotonic job ID
        self._active_job_id = -1  # job ID of data currently in ghost_buf

        # Multi-variant ghost lookups (different context windows)
        self._num_ghost_variants = getattr(server_args, 'phantom_num_ghosts', 1)

        # Negative filter (bloom filter for rejected bigrams)
        self._neg_filter = _NegativeFilter()

        # Adaptive controller for negative filter pipeline (phases 2+3)
        # "threshold" here controls bloom filter age_limit:
        #   lower threshold → keep more history (aggressive filtering)
        #   higher threshold → age faster (conservative, less filtering)
        # Range: 500 (very aggressive) to 5000 (nearly disabled)
        self._neg_controller = AdaptiveThresholdController(
            initial_threshold=2000.0,
            min_threshold=500.0,
            max_threshold=5000.0,
            ema_alpha=0.1,
            precision_target=0.70,   # 70% of patches should be correct
            precision_floor=0.30,    # below 30% → disable scan+patch
            warmup_steps=20,         # passthrough for 20 rounds while bloom fills
            step_size=200.0,         # adjust age_limit by 200 per step
            backoff_cooldown=40,     # stay disabled for 40 rounds before retry
        )
        # Track which positions were patched this round for outcome measurement
        self._patched_positions: list = []  # [(request_idx, position_idx), ...]

        # Quantization telemetry — collects signals for offline calibration
        # Snapshots written to /tmp/phantom_quant_telem/ by default
        telem_dir = os.environ.get(
            "PHANTOM_QUANT_TELEM_DIR", "/tmp/phantom_quant_telem"
        )
        self._quant_telem = _QuantTelemetry(
            snapshot_interval=500,
            snapshot_dir=telem_dir,
        )

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
        self._neg_filter.reset()
        self._neg_controller = AdaptiveThresholdController(
            initial_threshold=2000.0,
            min_threshold=500.0,
            max_threshold=5000.0,
            ema_alpha=0.1,
            precision_target=0.70,
            precision_floor=0.30,
            warmup_steps=20,
            step_size=200.0,
            backoff_cooldown=40,
        )
        self._patched_positions = []
        self._quant_telem.reset()
        self._job_counter = 0
        self._active_job_id = -1

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
        """Background CPU thread: 3-phase write → scan → patch pipeline.

        Runs entirely while the GPU is busy verifying the current batch:

          Phase 1 (WRITE):  corpus lookup → initial draft into numpy arrays
          Phase 2 (SCAN):   bloom filter identifies bad bigram positions
          Phase 3 (PATCH):  re-query corpus with alt context windows,
                            cherry-pick token replacements for bad positions

        By the time the GPU needs the next draft, all 3 phases have completed
        and the patched result is already in pinned HSA memory.

        When num_variants=1, the negative filter is cold (no data yet), or the
        adaptive controller has backed off, only Phase 1 runs — zero overhead
        vs the original single-lookup path.
        """
        while not self._ghost_stop.is_set():
            self._ghost_request_event.wait(timeout=0.1)
            if self._ghost_stop.is_set():
                break
            self._ghost_request_event.clear()

            with self._ghost_job_lock:
                job = self._ghost_job
                self._ghost_job = None

            if job is None:
                continue

            try:
                bs = job.bs
                K = job.K

                # ── Phase 1: WRITE — standard corpus lookup ──
                req_drafts, mask = self.corpus.batch_get(job.batch_tokens)
                if len(req_drafts) != bs * K:
                    continue

                # ── Phase 2 + 3: SCAN + PATCH (gated by adaptive controller) ──
                run_filter = (
                    job.num_variants > 1
                    and self._neg_filter.size > 0
                    and self._neg_controller.is_active
                )
                patched = []
                if run_filter:
                    bad_positions = self._scan_bad_positions(req_drafts, bs, K)
                    if bad_positions:
                        req_drafts, mask, patched = self._patch_bad_positions(
                            job, req_drafts, mask, bad_positions, bs, K
                        )
                self._patched_positions = patched

                # Copy final result into pinned buffer and signal ready
                cpu_tokens = self.ghost_buf.cpu_tokens
                cpu_mask = self.ghost_buf.cpu_mask
                cpu_tokens[:bs * K].copy_(torch.from_numpy(req_drafts))
                cpu_mask[:bs * K * K].copy_(torch.from_numpy(mask))

                self._active_job_id = job.job_id
                self.ghost_buf.swap(bs)

            except Exception as e:
                logger.debug("PHANTOM ghost build error: %s", e)

    def _scan_bad_positions(self, drafts: np.ndarray, bs: int, K: int) -> list:
        """Phase 2: identify draft positions containing known-bad bigrams.

        Returns list of (request_idx, position_idx) where the negative filter
        flagged a bad bigram. Position is the index of the second token in the
        bad pair (the one we'll try to replace).
        """
        bad = []
        for r in range(bs):
            seq = drafts[r * K: (r + 1) * K]
            for i in range(len(seq) - 1):
                tok_a, tok_b = int(seq[i]), int(seq[i + 1])
                if tok_a != 0 and tok_b != 0 and self._neg_filter.query_bigram(tok_a, tok_b):
                    bad.append((r, i + 1))
        return bad

    def _patch_bad_positions(self, job: _GhostJob, drafts: np.ndarray,
                             mask: np.ndarray, bad_positions: list,
                             bs: int, K: int):
        """Phase 3: cherry-pick replacement tokens for flagged positions.

        Re-queries the corpus with alternative context windows (wider, narrower).
        For each bad position, if the alt lookup produced a different token that
        doesn't itself form a known-bad bigram, swap it in. Mask rows are updated
        to match the alt lookup's tree structure at the patched position.

        Returns (drafts, mask, patched) where patched is a list of
        (request_idx, position_idx, original_token, replacement_token) tuples
        for outcome tracking and future contrastive training.
        """
        W = self.max_match_window_size
        alt_windows = [min(W * 2, 128), max(W // 2, 4)]
        patched = []  # (r, pos, orig_tok, new_tok)

        for win in alt_windows:
            alt_tokens = []
            for tokens in job.batch_tokens:
                alt_tokens.append(tokens[-win:] if len(tokens) > win else tokens)

            try:
                alt_drafts, alt_mask = self.corpus.batch_get(alt_tokens)
            except Exception:
                continue
            if len(alt_drafts) != bs * K:
                continue

            remaining = []
            for r, pos in bad_positions:
                alt_tok = int(alt_drafts[r * K + pos])
                orig_tok = int(drafts[r * K + pos])

                if alt_tok == 0 or alt_tok == orig_tok:
                    remaining.append((r, pos))
                    continue

                # Verify replacement doesn't create a new bad bigram
                prev_tok = int(drafts[r * K + pos - 1]) if pos > 0 else 0
                if prev_tok != 0 and self._neg_filter.query_bigram(prev_tok, alt_tok):
                    remaining.append((r, pos))
                    continue

                # Swap token and its mask row
                drafts[r * K + pos] = alt_tok
                m_off = r * K * K + pos * K
                mask[m_off:m_off + K] = alt_mask[m_off:m_off + K]
                patched.append((r, pos, orig_tok, alt_tok))

            bad_positions = remaining
            if not bad_positions:
                break

        return drafts, mask, patched

    def _submit_ghost_request(self, batch: ScheduleBatch):
        """Submit a _GhostJob for the ghost thread to process."""
        batch_tokens = []
        for req in batch.reqs:
            tokens = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_match_window_size
            )
            batch_tokens.append(tokens)

        self._job_counter += 1
        job = _GhostJob(
            job_id=self._job_counter,
            batch_tokens=batch_tokens,
            K=self.draft_token_num,
            bs=len(batch_tokens),
            num_variants=self._num_ghost_variants,
        )
        with self._ghost_job_lock:
            self._ghost_job = job
        self._ghost_request_event.set()

    # ---- Helpers ----

    @staticmethod
    def _efficient_concat_last_n(seq1: List[int], seq2: List[int], n: int):
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]
        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def _update_negative_filter(self, draft_tokens: torch.Tensor,
                                accept_lens: list, bs: int, K: int):
        """Insert rejected draft bigrams into the negative bloom filter.

        For each request, tokens beyond accept_length are "rejected" — the model
        disagreed with these drafts. We extract bigrams from the rejected suffix
        and insert them so future ghost variants can avoid these patterns.
        """
        try:
            draft_cpu = draft_tokens.cpu().numpy() if draft_tokens.is_cuda else draft_tokens.numpy()
            for r in range(bs):
                acc_len = int(accept_lens[r]) if r < len(accept_lens) else 0
                start = r * K + acc_len
                end = (r + 1) * K
                rejected = draft_cpu[start:end]
                # Insert bigrams from the rejected suffix
                for i in range(len(rejected) - 1):
                    tok_a, tok_b = int(rejected[i]), int(rejected[i + 1])
                    if tok_a != 0 and tok_b != 0:  # skip padding
                        self._neg_filter.insert_bigram(tok_a, tok_b)
        except Exception:
            pass  # non-critical — filter is best-effort

    def _record_patch_outcomes(self, accept_lens: list, K: int):
        """Compare patched positions against verify accept_lengths.

        A patch at position P in request R is "correct" if P < accept_length[R],
        meaning the replacement token was accepted by the target model.

        Feeds outcomes into the adaptive controller which adjusts:
          - Whether scan+patch phases should keep running
          - The bloom filter's age_limit (via threshold → age_limit sync)

        Also stores preference pairs for future contrastive training:
          (original_token, replacement_token, was_accepted)
        """
        patched = self._patched_positions
        if not patched:
            # No patches this round — still count the step so warmup advances
            self._neg_controller.record_outcome(0, 0)
            return

        n_patched = len(patched)
        n_correct = 0
        for r, pos, orig_tok, new_tok in patched:
            acc_len = int(accept_lens[r]) if r < len(accept_lens) else 0
            if pos < acc_len:
                n_correct += 1

        self._neg_controller.record_outcome(n_patched, n_correct)

        # Sync controller threshold → bloom filter age_limit
        # Lower threshold = keep more history = more aggressive filtering
        self._neg_filter._age_limit = max(int(self._neg_controller.threshold), 200)

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

        if (ghost_ready and self.ghost_buf.active_bs == bs
                and self._active_job_id == self._job_counter):
            # Ghost tree is ready and matches current job — read from pinned HSA memory
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

            # ── Feed rejected patterns into negative filter ──
            self._update_negative_filter(
                batch.spec_info.draft_token, accept_lens, bs, K
            )

            # ── Adaptive controller: measure patch outcomes ──
            self._record_patch_outcomes(accept_lens, K)

            # ── Quantization telemetry: capture free signals ──
            self._quant_telem.record(
                logits=logits_output.next_token_logits,
                hidden_states=logits_output.hidden_states,
                draft_tokens=batch.spec_info.draft_token,
                verified_ids=next_token_ids,
                accept_lens=accept_lens,
                bs=bs, K=K,
            )

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
        corpus_K = self._initial_draft_num  # corpus always returns this many

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

        # Corpus always returns corpus_K tokens per request; slice to current K
        if K < corpus_K:
            req_drafts_reshaped = req_drafts.reshape(bs, corpus_K)[:, :K].reshape(-1)
            mask_reshaped = mask.reshape(bs, corpus_K, corpus_K)[:, :K, :K].reshape(-1)
            req_drafts = np.ascontiguousarray(req_drafts_reshaped)
            mask = np.ascontiguousarray(mask_reshaped)
        elif K > corpus_K:
            # Dynamic γ grew beyond corpus K — pad with zeros
            padded = np.zeros(bs * K, dtype=req_drafts.dtype)
            padded_mask = np.zeros(bs * K * K, dtype=mask.dtype)
            for i in range(bs):
                padded[i * K:i * K + corpus_K] = req_drafts[i * corpus_K:(i + 1) * corpus_K]
            req_drafts = padded
            mask = padded_mask

        draft_tokens_gpu = torch.from_numpy(req_drafts).to(self.device, non_blocking=True)
        tree_mask_gpu = torch.from_numpy(mask).to(self.device, non_blocking=True)
        return draft_tokens_gpu, tree_mask_gpu

    def _fallback_target_only(self, batch: ScheduleBatch) -> GenerationBatchResult:
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = None
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
            neg_size = self._neg_filter.size if self._neg_filter else 0
            neg_age = self._neg_filter._count if self._neg_filter else 0
            ctrl = self._neg_controller.get_state()
            logger.info(
                "PHANTOM stats: rounds=%d, ghost=%.1f%% (%d/%d), "
                "accept=%.2f, γ=%d, corpus_hit=%.1f%%, fallback=%s, "
                "buf=%s, neg_filter=%d (age_lim=%d), ghosts=%d, "
                "patch_ema=%.2f τ=%.0f %s",
                self.total_rounds, ghost_pct, self.ghost_hits, total,
                avg_accept, self.draft_token_num, corpus_pct,
                "ON" if self._fallback_active else "off",
                "HSA" if self._is_hsa else "pinned",
                neg_size, self._neg_filter._age_limit,
                self._num_ghost_variants,
                ctrl["precision_ema"], ctrl["threshold"],
                "(warmup)" if ctrl["in_warmup"] else
                "(OFF)" if not ctrl["enabled"] else "",
            )
            # Quant telemetry summary (every 5th log interval = ~250 rounds)
            if self.total_rounds % (self._log_interval * 5) == 0:
                qt = self._quant_telem.get_summary()
                logger.info(
                    "PHANTOM quant telem: margin_acc=%.3f margin_rej=%.3f "
                    "confusion=%d ch_samples=%d pos_rate=%s",
                    qt["margin_acc"], qt["margin_rej"],
                    qt["confusion_pairs"], qt["channel_samples"],
                    qt["pos_rate"],
                )

    def __del__(self):
        try:
            self._stop_ghost_thread()
        except Exception:
            pass
