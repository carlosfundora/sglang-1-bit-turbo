"""
draft_prefilter.py — Layered pre-rejection filter for speculative draft tokens.

Eliminates draft candidates BEFORE expensive verification forward pass.
Three filter layers, cheapest first:

  L0: N-gram trie surprisal (CPU, ~zero cost)
      → if -log P(tok|context) > τ₀, DROP

  L1: Screen head inversion (INT8, fast GPU)
      → if screen is CONFIDENT about a token, DROP
        (screen is ~96% wrong — high confidence = noise signal)

  L2: Head agreement analysis (GPU, computed from existing logits)
      → Track inter-head divergence for telemetry + future fine-tuning
      → Unanimous agreement across heads → boost confidence
      → All heads disagree → mark as volatile

Adaptive thresholds:
  - Each filter layer has an independent EMA-tracked accuracy score
  - Thresholds tighten when filter is over-dropping (good tokens killed)
  - Thresholds loosen when filter is under-dropping (wasting verify compute)
  - Layer backs off entirely if its accuracy falls below a floor
  - Warm-up period: passthrough for first N steps while collecting data

Also collects per-position telemetry:
  - Head agreement matrix (which heads agree on which tokens)
  - Confidence spread (max_prob - min_prob across heads)
  - Surprisal histogram
  → Exported for contrastive fine-tuning of screen/volatile heads
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class FilterTelemetry:
    """Per-step telemetry for draft pre-filtering.

    Collected for offline analysis and contrastive fine-tuning.
    """
    # Counts
    total_candidates: int = 0
    ngram_dropped: int = 0
    screen_dropped: int = 0
    survivors: int = 0

    # Head agreement (per surviving position)
    head_agreement_scores: List[float] = field(default_factory=list)
    head_confidence_spreads: List[float] = field(default_factory=list)
    unanimous_positions: int = 0

    # For fine-tuning data collection
    # (position_idx, {head_i: token_id}, {head_i: confidence}, was_accepted)
    agreement_records: List[dict] = field(default_factory=list)

    @property
    def drop_rate(self) -> float:
        if self.total_candidates == 0:
            return 0.0
        return 1.0 - (self.survivors / self.total_candidates)

    def summary(self) -> str:
        return (
            f"candidates={self.total_candidates} "
            f"ngram_drop={self.ngram_dropped} "
            f"screen_drop={self.screen_dropped} "
            f"survivors={self.survivors} "
            f"drop_rate={self.drop_rate:.1%} "
            f"unanimous={self.unanimous_positions}"
        )


# ------------------------------------------------------------------
# Adaptive threshold controller
# ------------------------------------------------------------------


class AdaptiveThresholdController:
    """Self-tuning threshold for a single filter layer.

    Tracks whether drops were correct (dropped token would have been rejected
    by verify anyway) or harmful (dropped a token that would have been accepted).

    Uses exponential moving average (EMA) of precision to adjust:
      - precision high → tighten threshold (drop more aggressively)
      - precision low → loosen threshold (back off, dropping good tokens)
      - precision below floor → disable layer entirely until recovery

    The EMA window is short enough to react within ~20-50 steps but long
    enough to avoid oscillation from single-step noise.

    Args:
        initial_threshold: Starting threshold value
        min_threshold: Floor — never tighten beyond this
        max_threshold: Ceiling — never loosen beyond this (effectively disabled)
        ema_alpha: Smoothing factor (higher = more reactive, default 0.1)
        precision_target: Desired precision (fraction of drops that were correct)
        precision_floor: Below this, disable the layer entirely
        warmup_steps: Passthrough (no filtering) for this many steps
        step_size: How much to adjust threshold per update
        backoff_cooldown: Steps to wait before re-enabling after backoff
    """

    def __init__(
        self,
        initial_threshold: float,
        min_threshold: float,
        max_threshold: float,
        ema_alpha: float = 0.1,
        precision_target: float = 0.80,
        precision_floor: float = 0.40,
        warmup_steps: int = 30,
        step_size: float = 0.05,
        backoff_cooldown: int = 50,
    ):
        self.threshold = initial_threshold
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.ema_alpha = ema_alpha
        self.precision_target = precision_target
        self.precision_floor = precision_floor
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.backoff_cooldown = backoff_cooldown

        # State
        self.precision_ema: float = 0.5  # start neutral
        self.step_count: int = 0
        self.enabled: bool = True
        self._backoff_until: int = 0  # step at which to re-enable
        self._recent_drops: deque = deque(maxlen=200)  # (was_correct, value) pairs
        self._total_correct_drops: int = 0
        self._total_wrong_drops: int = 0

    @property
    def is_active(self) -> bool:
        """Whether this layer should filter (past warmup, not backed off)."""
        if self.step_count < self.warmup_steps:
            return False
        if not self.enabled and self.step_count < self._backoff_until:
            return False
        if not self.enabled and self.step_count >= self._backoff_until:
            # Cooldown expired — cautiously re-enable at loosest threshold
            self.enabled = True
            self.threshold = self.max_threshold * 0.9  # re-enter conservatively
            self.precision_ema = 0.5  # reset EMA
            logger.info(
                "AdaptiveThreshold: re-enabling after backoff, τ=%.3f",
                self.threshold,
            )
        return self.enabled

    def record_outcome(self, n_dropped: int, n_correct_drops: int):
        """Record how many drops were correct after verification.

        Args:
            n_dropped: Total tokens dropped by this layer this step
            n_correct_drops: How many of those would have been rejected anyway
                             (i.e., the drop was correct — we saved compute)
        """
        self.step_count += 1

        if n_dropped == 0:
            return

        step_precision = n_correct_drops / n_dropped
        self.precision_ema = (
            self.ema_alpha * step_precision
            + (1 - self.ema_alpha) * self.precision_ema
        )
        self._total_correct_drops += n_correct_drops
        self._total_wrong_drops += (n_dropped - n_correct_drops)

        # Store for recent window analysis
        for _ in range(n_correct_drops):
            self._recent_drops.append(True)
        for _ in range(n_dropped - n_correct_drops):
            self._recent_drops.append(False)

        # Adjust threshold based on EMA precision
        self._adjust()

    def _adjust(self):
        """Dial threshold based on precision EMA."""
        if self.step_count < self.warmup_steps:
            return

        # Below floor → back off entirely
        if self.precision_ema < self.precision_floor:
            self.enabled = False
            self._backoff_until = self.step_count + self.backoff_cooldown
            logger.warning(
                "AdaptiveThreshold: BACKING OFF — precision=%.2f < floor=%.2f, "
                "disabled for %d steps (τ was %.3f)",
                self.precision_ema,
                self.precision_floor,
                self.backoff_cooldown,
                self.threshold,
            )
            return

        # Above target → tighten (drop more aggressively, lower threshold)
        if self.precision_ema > self.precision_target + 0.05:
            new_t = self.threshold - self.step_size
            self.threshold = max(new_t, self.min_threshold)

        # Below target → loosen (drop less, raise threshold)
        elif self.precision_ema < self.precision_target - 0.05:
            new_t = self.threshold + self.step_size
            self.threshold = min(new_t, self.max_threshold)

        # Within ±0.05 of target → fine-tune with half step
        elif abs(self.precision_ema - self.precision_target) > 0.02:
            direction = -1 if self.precision_ema > self.precision_target else 1
            new_t = self.threshold + direction * self.step_size * 0.3
            self.threshold = max(self.min_threshold, min(new_t, self.max_threshold))

    def get_state(self) -> dict:
        """Snapshot for logging/monitoring."""
        recent_prec = (
            sum(self._recent_drops) / len(self._recent_drops)
            if self._recent_drops
            else 0.0
        )
        return {
            "threshold": round(self.threshold, 4),
            "precision_ema": round(self.precision_ema, 4),
            "recent_precision": round(recent_prec, 4),
            "enabled": self.enabled,
            "step": self.step_count,
            "in_warmup": self.step_count < self.warmup_steps,
            "total_correct_drops": self._total_correct_drops,
            "total_wrong_drops": self._total_wrong_drops,
        }


class DraftPreFilter:
    """Layered pre-rejection filter for Medusa draft tokens.

    Designed to slot between draft generation and tree verification
    in MedusaWorker._forward_medusa.

    Each filter layer has an independent AdaptiveThresholdController that:
      - Warms up for N steps (passthrough, collecting baseline data)
      - Tracks drop precision via EMA (were drops correct?)
      - Tightens threshold when over-dropping good tokens
      - Loosens threshold when under-dropping garbage
      - Backs off entirely if precision falls below a floor
      - Re-enables cautiously after cooldown with loosest threshold

    Args:
        ngram_trie: Optional NgramTrie for L0 surprisal filtering
        screen_head_idx: Index of the screen head in MedusaModel (default: None = no screen)
        surprisal_threshold: L0 initial drop threshold (-log P > τ → drop)
        screen_confidence_threshold: L1 initial drop threshold (screen P(tok) > τ → drop)
        collect_telemetry: Whether to collect detailed agreement records
        telemetry_buffer_size: Max records to keep in memory before flush
    """

    def __init__(
        self,
        ngram_trie=None,
        screen_head_idx: Optional[int] = None,
        surprisal_threshold: float = 8.0,
        screen_confidence_threshold: float = 0.3,
        collect_telemetry: bool = True,
        telemetry_buffer_size: int = 10000,
    ):
        self.ngram_trie = ngram_trie
        self.screen_head_idx = screen_head_idx
        self.collect_telemetry = collect_telemetry
        self.telemetry_buffer_size = telemetry_buffer_size

        # Adaptive controllers for L0 and L1
        # L0 N-gram: surprisal range ~0 (certain) to ~17 (vocab-uniform)
        self.ngram_controller = AdaptiveThresholdController(
            initial_threshold=surprisal_threshold,
            min_threshold=3.0,       # very aggressive — drop anything above 3 bits
            max_threshold=14.0,      # very conservative — almost never drop
            ema_alpha=0.08,
            precision_target=0.80,
            precision_floor=0.35,
            warmup_steps=30,
            step_size=0.4,           # surprisal scale is ~0-17, steps of 0.4
            backoff_cooldown=60,
        )

        # L1 Screen: confidence range 0.0 (unsure) to 1.0 (certain)
        self.screen_controller = AdaptiveThresholdController(
            initial_threshold=screen_confidence_threshold,
            min_threshold=0.05,      # very aggressive — drop if screen >5% confident
            max_threshold=0.85,      # very conservative — only drop near-certain
            ema_alpha=0.10,
            precision_target=0.75,
            precision_floor=0.30,
            warmup_steps=30,
            step_size=0.03,          # confidence scale is 0-1, steps of 0.03
            backoff_cooldown=60,
        )

        # Track which tokens were dropped (for post-verify feedback)
        self._last_drop_info: Optional[dict] = None

        # Running telemetry accumulator
        self._telemetry_buffer: List[FilterTelemetry] = []
        self._lifetime_stats = {
            "total_candidates": 0,
            "total_ngram_dropped": 0,
            "total_screen_dropped": 0,
            "total_survivors": 0,
            "total_unanimous": 0,
            "steps": 0,
        }

        # Agreement data for contrastive fine-tuning export
        self._agreement_data: List[dict] = []

        logger.info(
            "DraftPreFilter: L0=%s(τ₀=%.1f) L1=%s(τ₁=%.2f) adaptive=ON warmup=30",
            "ngram" if ngram_trie else "off",
            surprisal_threshold,
            f"screen[{screen_head_idx}]" if screen_head_idx is not None else "off",
            screen_confidence_threshold,
            collect_telemetry,
        )

    # ------------------------------------------------------------------
    # Main filter entry point
    # ------------------------------------------------------------------

    def filter_drafts(
        self,
        draft_tokens: torch.Tensor,
        head_logits: List[torch.Tensor],
        context_ids: Optional[List[List[int]]] = None,
        screen_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, FilterTelemetry]:
        """Filter draft tokens through all active layers.

        Layers that are in warmup, backed off, or disabled are skipped.
        Adaptive thresholds are used instead of fixed values.

        Args:
            draft_tokens: [bs, K] draft token IDs from Medusa heads
            head_logits: List of [bs, vocab] logits per head
            context_ids: Per-request context token lists for n-gram lookup
            screen_logits: [bs, vocab] logits from screen head (if separate)

        Returns:
            filtered_tokens: [bs, K] with dropped tokens replaced by -1
            keep_mask: [bs, K] bool tensor (True = keep, False = dropped)
            telemetry: FilterTelemetry for this step
        """
        bs, K = draft_tokens.shape
        keep_mask = torch.ones(bs, K, dtype=torch.bool, device=draft_tokens.device)
        telem = FilterTelemetry(total_candidates=bs * K)

        # Track which layer dropped which positions (for post-verify feedback)
        ngram_dropped_mask = torch.zeros_like(keep_mask)
        screen_dropped_mask = torch.zeros_like(keep_mask)

        # L0: N-gram surprisal (only if controller says active)
        if (
            self.ngram_trie is not None
            and context_ids is not None
            and self.ngram_controller.is_active
        ):
            ngram_mask = self._filter_ngram(
                draft_tokens, context_ids, bs, K,
                threshold=self.ngram_controller.threshold,
            )
            ngram_dropped_mask = ~ngram_mask
            dropped = ngram_dropped_mask.sum().item()
            telem.ngram_dropped = dropped
            keep_mask &= ngram_mask

        # L1: Screen head inversion (only if controller says active)
        if (
            screen_logits is not None
            and self.screen_controller.is_active
        ):
            screen_mask = self._filter_screen(
                draft_tokens, screen_logits, keep_mask, bs, K,
                threshold=self.screen_controller.threshold,
            )
            screen_dropped_mask = keep_mask & ~screen_mask
            dropped = screen_dropped_mask.sum().item()
            telem.screen_dropped = dropped
            keep_mask &= screen_mask

        # L2: Head agreement analysis (always runs — telemetry + boosting)
        agreement_info = self._analyze_head_agreement(
            draft_tokens, head_logits, keep_mask, bs, K
        )
        telem.head_agreement_scores = agreement_info["scores"]
        telem.head_confidence_spreads = agreement_info["spreads"]
        telem.unanimous_positions = agreement_info["unanimous"]
        if self.collect_telemetry:
            telem.agreement_records = agreement_info["records"]

        telem.survivors = keep_mask.sum().item()

        # Apply mask: replace dropped tokens with -1 sentinel
        filtered = draft_tokens.clone()
        filtered[~keep_mask] = -1

        # Store drop info for post-verify feedback
        self._last_drop_info = {
            "draft_tokens": draft_tokens.clone(),
            "ngram_dropped": ngram_dropped_mask.clone(),
            "screen_dropped": screen_dropped_mask.clone(),
            "keep_mask": keep_mask.clone(),
        }

        # Update lifetime stats
        self._update_stats(telem)

        return filtered, keep_mask, telem

    def record_verify_results(self, accepted_mask: torch.Tensor):
        """Feed verification results back to adaptive controllers.

        Called AFTER tree verification with a mask of which draft positions
        were actually accepted by the target model. This closes the feedback
        loop: we can now tell each layer whether its drops were correct.

        A drop was CORRECT if the token would have been rejected anyway
        (i.e., accepted_mask[pos] == False for that position).
        A drop was WRONG if the token would have been accepted
        (i.e., accepted_mask[pos] == True — we killed a good token).

        Args:
            accepted_mask: [bs, K] bool — True if verify accepted this position
        """
        if self._last_drop_info is None:
            return

        info = self._last_drop_info
        ngram_dropped = info["ngram_dropped"]
        screen_dropped = info["screen_dropped"]

        # For each layer: count how many drops were correct
        # Correct drop = token was dropped AND would NOT have been accepted
        if ngram_dropped.any():
            n_ngram_dropped = ngram_dropped.sum().item()
            # A "correct" drop: we dropped it, and verify also wouldn't accept it
            n_correct = (ngram_dropped & ~accepted_mask).sum().item()
            self.ngram_controller.record_outcome(n_ngram_dropped, n_correct)

        if screen_dropped.any():
            n_screen_dropped = screen_dropped.sum().item()
            n_correct = (screen_dropped & ~accepted_mask).sum().item()
            self.screen_controller.record_outcome(n_screen_dropped, n_correct)

        self._last_drop_info = None

    # ------------------------------------------------------------------
    # L0: N-gram surprisal filter
    # ------------------------------------------------------------------

    def _filter_ngram(
        self,
        draft_tokens: torch.Tensor,
        context_ids: List[List[int]],
        bs: int,
        K: int,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """Drop tokens with high n-gram surprisal.

        Surprisal = -log₂(P(token | context))
        If surprisal > threshold → token is extremely unlikely given context.
        """
        τ = threshold if threshold is not None else self.ngram_controller.threshold
        mask = torch.ones(bs, K, dtype=torch.bool, device=draft_tokens.device)
        draft_cpu = draft_tokens.cpu().tolist()

        for b in range(bs):
            ctx = context_ids[b] if b < len(context_ids) else []
            for k in range(K):
                tok = draft_cpu[b][k]
                # Build extended context: original context + preceding drafts
                extended_ctx = ctx + draft_cpu[b][:k]
                prob = self.ngram_trie.confidence(extended_ctx, tok)

                if prob > 0:
                    surprisal = -math.log2(prob)
                    if surprisal > τ:
                        mask[b, k] = False
                else:
                    # Never seen this n-gram — don't drop (no evidence)
                    # Could apply Laplace smoothing here for stricter filtering
                    pass

        return mask

    # ------------------------------------------------------------------
    # L1: Screen head inversion filter
    # ------------------------------------------------------------------

    def _filter_screen(
        self,
        draft_tokens: torch.Tensor,
        screen_logits: torch.Tensor,
        current_mask: torch.Tensor,
        bs: int,
        K: int,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """Drop tokens the screen head is confident about.

        The screen head is a quantized broken clone (~96% wrong).
        If it assigns high probability to a token, that token is likely noise.
        Inverted logic: screen confidence → our suspicion.
        """
        τ = threshold if threshold is not None else self.screen_controller.threshold
        mask = torch.ones(bs, K, dtype=torch.bool, device=draft_tokens.device)

        # Softmax the screen logits to get probabilities
        screen_probs = F.softmax(screen_logits, dim=-1)  # [bs, vocab]

        for b in range(bs):
            if not current_mask[b].any():
                continue
            for k in range(K):
                if not current_mask[b, k]:
                    continue
                tok = draft_tokens[b, k].item()
                screen_conf = screen_probs[b, tok].item()

                # Screen is confidently predicting this token → suspicious
                if screen_conf > τ:
                    mask[b, k] = False

        return mask

    # ------------------------------------------------------------------
    # L2: Head agreement analysis
    # ------------------------------------------------------------------

    def _analyze_head_agreement(
        self,
        draft_tokens: torch.Tensor,
        head_logits: List[torch.Tensor],
        keep_mask: torch.Tensor,
        bs: int,
        K: int,
    ) -> dict:
        """Analyze agreement/divergence across Medusa heads.

        For each draft position, check:
        1. Do multiple heads agree on the same token? (unanimous = high confidence)
        2. How spread out are the head confidences? (spread = uncertainty)
        3. How close/far are heads from each other? (KL divergence between pairs)

        Returns telemetry dict with scores, spreads, unanimity count, and
        per-position records for fine-tuning data collection.
        """
        n_heads = len(head_logits)
        scores = []
        spreads = []
        records = []
        unanimous = 0

        # Get per-head top-1 tokens and confidences
        head_top_tokens = []  # [n_heads, bs]
        head_top_confs = []   # [n_heads, bs]
        for h_logits in head_logits:
            probs = F.softmax(h_logits, dim=-1)  # [bs, vocab]
            top_conf, top_tok = probs.max(dim=-1)  # [bs], [bs]
            head_top_tokens.append(top_tok)
            head_top_confs.append(top_conf)

        head_top_tokens_t = torch.stack(head_top_tokens, dim=0)  # [n_heads, bs]
        head_top_confs_t = torch.stack(head_top_confs, dim=0)    # [n_heads, bs]

        for b in range(bs):
            for k in range(K):
                if not keep_mask[b, k]:
                    continue

                # Which heads predict the same token as head k?
                target_tok = draft_tokens[b, k].item()
                agreements = 0
                head_predictions = {}
                head_confidences = {}

                for h in range(n_heads):
                    h_tok = head_top_tokens_t[h, b].item()
                    h_conf = head_top_confs_t[h, b].item()
                    head_predictions[h] = h_tok
                    head_confidences[h] = h_conf
                    if h_tok == target_tok:
                        agreements += 1

                # Agreement score: fraction of heads that agree
                agreement_score = agreements / n_heads
                scores.append(agreement_score)

                # Confidence spread: max - min confidence across heads
                confs = [head_confidences[h] for h in range(n_heads)]
                spread = max(confs) - min(confs)
                spreads.append(spread)

                # Unanimous = all heads agree on same token
                unique_predictions = len(set(head_predictions.values()))
                if unique_predictions == 1:
                    unanimous += 1

                # Collect record for fine-tuning
                if self.collect_telemetry:
                    records.append({
                        "batch_idx": b,
                        "draft_pos": k,
                        "target_token": target_tok,
                        "head_predictions": head_predictions,
                        "head_confidences": head_confidences,
                        "agreement_score": agreement_score,
                        "confidence_spread": spread,
                        "n_unique_predictions": unique_predictions,
                    })

        return {
            "scores": scores,
            "spreads": spreads,
            "unanimous": unanimous,
            "records": records,
        }

    # ------------------------------------------------------------------
    # Telemetry management
    # ------------------------------------------------------------------

    def _update_stats(self, telem: FilterTelemetry):
        """Update lifetime statistics."""
        s = self._lifetime_stats
        s["total_candidates"] += telem.total_candidates
        s["total_ngram_dropped"] += telem.ngram_dropped
        s["total_screen_dropped"] += telem.screen_dropped
        s["total_survivors"] += telem.survivors
        s["total_unanimous"] += telem.unanimous_positions
        s["steps"] += 1

        if self.collect_telemetry:
            self._agreement_data.extend(telem.agreement_records)
            # Trim buffer if needed
            if len(self._agreement_data) > self.telemetry_buffer_size:
                self._agreement_data = self._agreement_data[
                    -self.telemetry_buffer_size :
                ]

    def get_stats(self) -> dict:
        """Get lifetime filter statistics including adaptive threshold state."""
        s = self._lifetime_stats
        total = s["total_candidates"] or 1
        return {
            **s,
            "ngram_drop_rate": s["total_ngram_dropped"] / total,
            "screen_drop_rate": s["total_screen_dropped"] / total,
            "overall_drop_rate": 1.0 - (s["total_survivors"] / total),
            "unanimity_rate": s["total_unanimous"] / (s["total_survivors"] or 1),
            "adaptive": {
                "ngram": self.ngram_controller.get_state(),
                "screen": self.screen_controller.get_state(),
            },
        }

    def export_agreement_data(self) -> List[dict]:
        """Export collected agreement records for contrastive fine-tuning.

        Each record contains:
          - head_predictions: {head_idx: token_id}
          - head_confidences: {head_idx: float}
          - agreement_score: float [0, 1]
          - confidence_spread: float
          - n_unique_predictions: int

        Use cases:
          1. Find positions where heads strongly agree → positive training signal
          2. Find positions where heads maximally disagree → contrastive training
          3. Find tokens the screen head was confident about → anti-examples
          4. Calibrate surprisal/confidence thresholds from real data
        """
        data = list(self._agreement_data)
        self._agreement_data.clear()
        return data

    def log_periodic_stats(self, interval: int = 100):
        """Log stats every N steps including adaptive threshold state."""
        if self._lifetime_stats["steps"] % interval == 0:
            stats = self.get_stats()
            adaptive = stats["adaptive"]
            ngram_s = adaptive["ngram"]
            screen_s = adaptive["screen"]
            logger.info(
                "DraftPreFilter [%d steps]: drop=%.1f%% "
                "(ngram=%.1f%% screen=%.1f%%) unanimity=%.1f%% | "
                "L0: τ=%.2f prec=%.2f %s | "
                "L1: τ=%.3f prec=%.2f %s",
                stats["steps"],
                stats["overall_drop_rate"] * 100,
                stats["ngram_drop_rate"] * 100,
                stats["screen_drop_rate"] * 100,
                stats["unanimity_rate"] * 100,
                ngram_s["threshold"],
                ngram_s["precision_ema"],
                "WARMUP" if ngram_s["in_warmup"] else (
                    "OFF" if not ngram_s["enabled"] else "ON"
                ),
                screen_s["threshold"],
                screen_s["precision_ema"],
                "WARMUP" if screen_s["in_warmup"] else (
                    "OFF" if not screen_s["enabled"] else "ON"
                ),
            )
