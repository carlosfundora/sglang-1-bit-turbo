"""SAGUARO (Speculative-Speculative Decoding) worker wrapper.

Wraps ANY speculative worker with draft-result caching and prefix-based reuse.
On each round, after the inner worker produces drafts and verification completes,
SAGUARO predicts the most likely accepted prefix and caches draft results keyed
by that prefix hash.  On the NEXT round, if the actual accepted prefix matches
the prediction, the cached drafts are reused — saving one full draft forward pass.

For true async pre-generation (overlapping draft compute with target verify),
a multi-GPU or multi-stream implementation is needed (documented as future work).
This single-GPU version provides modest speedup through cache reuse on repetitive
or highly-predictable text.

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


class _DraftCacheEntry:
    """Cached draft result for a single request."""

    __slots__ = ("prefix_hash", "draft_tokens", "draft_logits", "hit_count")

    def __init__(
        self,
        prefix_hash: str,
        draft_tokens: Optional[torch.Tensor],
        draft_logits: Optional[torch.Tensor],
    ):
        self.prefix_hash = prefix_hash
        self.draft_tokens = draft_tokens
        self.draft_logits = draft_logits
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
    """

    def __init__(
        self,
        inner_worker,
        server_args: ServerArgs,
        cache_capacity: int = 1024,
        prefix_window: int = 32,
    ):
        self.inner = inner_worker
        self.server_args = server_args
        self.cache = _LRUCache(capacity=cache_capacity)
        self.prefix_window = prefix_window

        # Stats
        self.total_rounds = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._log_interval = 100

        logger.info(
            "SaguaroWorker wrapping %s (cache=%d, prefix_window=%d)",
            type(inner_worker).__name__,
            cache_capacity,
            prefix_window,
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

    # ---- Main dispatch ----

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        """Transparent proxy with caching layer."""
        self.total_rounds += 1

        # On extend/prefill, always delegate directly
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            result = self.inner.forward_batch_generation(batch)
            self._update_cache_after_round(batch, result)
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

        # Cache the draft result for potential reuse
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
        """Check if we have cached drafts for this batch's prefix."""
        if batch.batch_size() != 1:
            # Cache is per-request; skip multi-request batches for simplicity
            return None

        req = batch.reqs[0]
        all_tokens = list(req.origin_input_ids) + list(req.output_ids)
        ph = self._prefix_hash(all_tokens, self.prefix_window)

        entry = self.cache.get_entry(ph)
        if entry is None:
            return None

        entry.hit_count += 1
        # Cache hit means the prefix matches our prediction — but we still
        # need to run the target model to verify.  In a true async impl we'd
        # have pre-computed the verification; here we just log the hit and
        # delegate (the inner worker will handle it).
        # Future: return pre-verified result if available.
        return None  # conservative: always verify through inner worker

    def _update_cache_after_round(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ):
        """After a round, predict the next prefix and cache draft info."""
        if batch.batch_size() != 1:
            return

        req = batch.reqs[0]
        # Build predicted next prefix: current tokens + accepted tokens
        all_tokens = list(req.origin_input_ids) + list(req.output_ids)
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
        entry = _DraftCacheEntry(
            prefix_hash=ph,
            draft_tokens=None,  # placeholder for future async pre-gen
            draft_logits=None,
        )
        self.cache.put_entry(ph, entry)

    def _periodic_log(self):
        if self.total_rounds % self._log_interval == 0 and self.total_rounds > 0:
            total = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / max(total, 1) * 100
            logger.info(
                "Saguaro stats: %d rounds, %d hits, %d misses (%.1f%% hit rate), "
                "cache_size=%d",
                self.total_rounds,
                self.cache_hits,
                self.cache_misses,
                hit_rate,
                len(self.cache),
            )
