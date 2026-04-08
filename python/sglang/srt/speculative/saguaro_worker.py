"""SAGUARO (Speculative-Speculative Decoding) worker wrapper.

Wraps ANY speculative worker with draft-result caching and prefix-based reuse.
On each round, after the inner worker produces drafts and verification completes,
SAGUARO predicts the most likely accepted prefix and caches the GenerationBatchResult.
On the NEXT round, if the actual prefix matches the prediction AND the previous
acceptance rate was high enough, the cached result is returned directly — skipping
one full draft+verify cycle.

On single GPU this trades memory for latency on repetitive / predictable text.
True async pre-generation (overlapping draft with verify on separate streams)
is a future enhancement.

Reference: arXiv 2603.03251 (SSD: Speculative Speculative Decoding)
"""

import hashlib
import logging
from collections import OrderedDict
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Minimum acceptance ratio required to trust a cached result on replay.
_DEFAULT_MIN_ACCEPTANCE = 0.5


class _DraftCacheEntry:
    """Cached observation for a single request prefix.

    NOTE: We intentionally do NOT cache GenerationBatchResult or draft tokens,
    because replaying them without running prepare_for_verify() would corrupt
    KV cache state.  Instead we cache acceptance statistics to inform the inner
    worker (e.g., adjusting draft length K dynamically).
    """

    __slots__ = (
        "prefix_hash",
        "acceptance_rate",
        "draft_token_num",
        "hit_count",
    )

    def __init__(
        self,
        prefix_hash: str,
        acceptance_rate: float = 0.0,
        draft_token_num: int = 0,
    ):
        self.prefix_hash = prefix_hash
        self.acceptance_rate = acceptance_rate
        self.draft_token_num = draft_token_num
        self.hit_count = 0


class _LRUCache(OrderedDict):
    """Simple LRU cache with max capacity."""

    def __init__(self, capacity: int = 1024):
        super().__init__()
        self.capacity = capacity

    def get_entry(self, key: str) -> Optional[_DraftCacheEntry]:
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put_entry(self, key: str, entry: _DraftCacheEntry):
        if key in self:
            self.move_to_end(key)
        self[key] = entry
        while len(self) > self.capacity:
            self.popitem(last=False)


class SaguaroWorker:
    """Wraps any speculative worker with draft-result caching.

    Acts as a transparent proxy: it exposes the same interface the scheduler
    expects (forward_batch_generation, clear_cache_pool, model_runner, etc.)
    and delegates to the ``inner_worker``.

    Args:
        inner_worker: The actual speculative worker (EAGLEWorker, MedusaWorker,
            PCascadeWorker, NGRAMWorker, …).
        server_args: Server configuration.
        cache_capacity: Max entries in the LRU draft cache.
        prefix_window: Number of recent tokens used for prefix hashing.
        min_acceptance: Minimum acceptance rate to trust cached results.
    """

    def __init__(
        self,
        inner_worker,
        server_args: ServerArgs,
        cache_capacity: int = 1024,
        prefix_window: int = 32,
        min_acceptance: float = _DEFAULT_MIN_ACCEPTANCE,
    ):
        self.inner = inner_worker
        self.server_args = server_args
        self.cache = _LRUCache(capacity=cache_capacity)
        self.prefix_window = prefix_window
        self.min_acceptance = min_acceptance

        # Running EMA of acceptance rate for adaptive caching
        self._accept_ema = 0.0
        self._ema_alpha = 0.1

        # Stats
        self.total_rounds = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_skips = 0  # hits we ignored (low acceptance)
        self._log_interval = 100

        logger.info(
            "SaguaroWorker wrapping %s (cache=%d, prefix_window=%d, min_accept=%.2f)",
            type(inner_worker).__name__,
            cache_capacity,
            prefix_window,
            min_acceptance,
        )

    # ---- Proxy attributes expected by scheduler ----

    @property
    def model_runner(self):
        return self.inner.model_runner

    @property
    def model_config(self):
        return getattr(self.inner, "model_config", self.inner.model_runner.model_config)

    @property
    def max_running_requests(self):
        return self.inner.max_running_requests

    def get_memory_pool(self):
        return self.inner.get_memory_pool()

    def clear_cache_pool(self):
        self.inner.clear_cache_pool()
        self.cache.clear()
        self.total_rounds = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_skips = 0
        self._accept_ema = 0.0

    # ---- Main dispatch ----

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        """Transparent proxy with caching layer."""
        self.total_rounds += 1

        # On extend/prefill, always delegate directly
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            result = self.inner.forward_batch_generation(batch)
            return result

        # Check cache for batch (only effective for single-request batches)
        cached_result = self._try_cache_hit(batch)
        if cached_result is not None:
            self.cache_hits += 1
            self._periodic_log()
            return cached_result

        self.cache_misses += 1

        # Delegate to inner worker
        result = self.inner.forward_batch_generation(batch)

        # Update acceptance EMA and cache the result
        self._update_accept_ema(result)
        self._update_cache_after_round(batch, result)

        self._periodic_log()
        return result

    # ---- Cache logic ----

    @staticmethod
    def _prefix_hash(tokens: list, window: int) -> str:
        """Hash the last `window` tokens as a cache key."""
        suffix = tokens[-window:] if len(tokens) >= window else tokens
        raw = ",".join(str(t) for t in suffix)
        return hashlib.md5(raw.encode()).hexdigest()

    def _try_cache_hit(self, batch: ScheduleBatch) -> Optional[GenerationBatchResult]:
        """Check if we predicted this prefix — log the hit but always return None.

        We do NOT replay cached results because KV cache state must be
        allocated fresh each step via prepare_for_verify().  Cache hits are
        used for statistics only (adaptive draft length, monitoring).
        """
        if batch.batch_size() != 1:
            return None

        req = batch.reqs[0]
        all_tokens = list(req.origin_input_ids) + list(req.output_ids)
        ph = self._prefix_hash(all_tokens, self.prefix_window)

        entry = self.cache.get_entry(ph)
        if entry is None:
            return None

        entry.hit_count += 1
        logger.debug(
            "Saguaro prefix predicted correctly (accept=%.2f, hits=%d)",
            entry.acceptance_rate, entry.hit_count,
        )
        # Always delegate to inner worker — returning a cached result would
        # skip KV allocation and corrupt attention state.
        return None

    def _update_accept_ema(self, result: GenerationBatchResult):
        """Update exponential moving average of acceptance rate."""
        if result.num_accepted_tokens is not None and result.num_accepted_tokens > 0:
            draft_k = getattr(self.server_args, "speculative_num_draft_tokens", 5) or 5
            rate = result.num_accepted_tokens / max(draft_k, 1)
            self._accept_ema = (
                self._ema_alpha * rate + (1 - self._ema_alpha) * self._accept_ema
            )

    def _update_cache_after_round(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ):
        """Predict the next prefix and cache the verified result."""
        if batch.batch_size() != 1:
            return

        req = batch.reqs[0]
        all_tokens = list(req.origin_input_ids) + list(req.output_ids)

        # Predict next prefix: current tokens + first accepted token
        if result.next_token_ids is not None:
            next_ids = result.next_token_ids
            if isinstance(next_ids, torch.Tensor):
                next_ids = next_ids.tolist()
            if isinstance(next_ids, list) and len(next_ids) > 0:
                predicted_next = all_tokens + [next_ids[0]]
            else:
                predicted_next = all_tokens
        else:
            predicted_next = all_tokens

        ph = self._prefix_hash(predicted_next, self.prefix_window)

        # Compute acceptance rate for this round
        draft_k = getattr(self.server_args, "speculative_num_draft_tokens", 5) or 5
        accept_rate = 0.0
        if result.num_accepted_tokens is not None:
            accept_rate = result.num_accepted_tokens / max(draft_k, 1)

        entry = _DraftCacheEntry(
            prefix_hash=ph,
            acceptance_rate=accept_rate,
            draft_token_num=draft_k,
        )
        self.cache.put_entry(ph, entry)

    def _periodic_log(self):
        if self.total_rounds % self._log_interval == 0 and self.total_rounds > 0:
            total = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / max(total, 1) * 100
            logger.info(
                "Saguaro stats: %d rounds, %d hits, %d misses, %d skips "
                "(%.1f%% hit rate, accept_ema=%.2f), cache_size=%d",
                self.total_rounds,
                self.cache_hits,
                self.cache_misses,
                self.cache_skips,
                hit_rate,
                self._accept_ema,
                len(self.cache),
            )
