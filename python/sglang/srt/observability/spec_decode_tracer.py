"""
spec_decode_tracer.py — Phase-boundary instrumentation for speculative decoding.

Enabled via SGLANG_SPEC_DIAG=1. Emits OTEL spans + structured JSON logs
at every speculative decode phase transition to find where invariants break.

Traces: draft_start → draft_end → verify_start → verify_end → commit
Logs: per-request authoritative state at each boundary
Asserts: invariants that should hold across transitions

All output goes to both OTEL (→ Jaeger/Tempo) and Python logging (→ stdout/file).
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("sglang.spec_diag")

# Gate everything behind env var
ENABLED = os.environ.get("SGLANG_SPEC_DIAG", "0") == "1"
VERBOSE = os.environ.get("SGLANG_SPEC_DIAG_VERBOSE", "0") == "1"
FAIL_FAST = os.environ.get("SGLANG_SPEC_DIAG_FAIL_FAST", "0") == "1"

# OTEL tracer (lazy init)
_tracer = None
_otel_available = False


def _get_tracer():
    global _tracer, _otel_available
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        _tracer = trace.get_tracer("sglang.spec_decode_diag", "0.1.0")
        _otel_available = True
    except Exception:
        _tracer = None
        _otel_available = False
    return _tracer


@dataclass
class PhaseSnapshot:
    """Authoritative per-request state at a phase boundary."""
    phase: str
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    # Batch-level
    batch_size: int = 0
    forward_mode: str = ""
    # Per-request (parallel arrays)
    req_pool_indices: Optional[List[int]] = None
    seq_lens: Optional[List[int]] = None
    extend_seq_lens: Optional[List[int]] = None
    extend_prefix_lens: Optional[List[int]] = None
    positions: Optional[List[int]] = None  # first few positions per request
    # KV state
    kv_allocated: Optional[List[int]] = None
    # Speculative state
    drafted_tokens: Optional[List[int]] = None
    accepted_count: Optional[int] = None
    acceptance_rate: Optional[float] = None
    # Hidden state
    hidden_shape: Optional[str] = None
    hidden_norm: Optional[float] = None
    # Anomalies detected
    anomalies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in asdict(self).items():
            if v is not None and v != [] and v != "":
                d[k] = v
        return d


def _safe_tolist(t, max_items=16) -> Optional[List]:
    """Safely convert tensor to list, truncating for readability."""
    if t is None:
        return None
    try:
        if isinstance(t, torch.Tensor):
            flat = t.detach().cpu().flatten().tolist()
        elif hasattr(t, '__iter__'):
            flat = list(t)
        else:
            return [t]
        if len(flat) > max_items:
            return flat[:max_items] + [f"...+{len(flat)-max_items}"]
        return flat
    except Exception as e:
        return [f"<error: {e}>"]


def _tensor_stats(t) -> Dict[str, float]:
    """Quick tensor stats without GPU→CPU sync if possible."""
    if t is None or not isinstance(t, torch.Tensor):
        return {}
    try:
        return {
            "shape": str(list(t.shape)),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "norm": float(t.float().norm().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }
    except Exception:
        return {"shape": str(list(t.shape)) if hasattr(t, 'shape') else "?"}


class SpecDecodeTracer:
    """Singleton tracer for speculative decode phase boundaries."""

    def __init__(self):
        self._step_count = 0
        self._violations = []
        self._last_snapshot: Optional[PhaseSnapshot] = None

    def snapshot_forward_batch(
        self,
        phase: str,
        forward_batch,
        extra: Optional[Dict] = None,
    ) -> PhaseSnapshot:
        """Capture authoritative state from a ForwardBatch at a phase boundary."""
        if not ENABLED:
            return PhaseSnapshot(phase=phase)

        snap = PhaseSnapshot(phase=phase, batch_size=0)

        try:
            snap.forward_mode = str(getattr(forward_batch, 'forward_mode', '?'))
            bs = getattr(forward_batch, 'batch_size', 0)
            if hasattr(bs, 'item'):
                bs = bs.item()
            snap.batch_size = int(bs) if bs else 0

            snap.seq_lens = _safe_tolist(getattr(forward_batch, 'seq_lens', None))
            snap.extend_seq_lens = _safe_tolist(
                getattr(forward_batch, 'extend_seq_lens', None)
            )
            snap.extend_prefix_lens = _safe_tolist(
                getattr(forward_batch, 'extend_prefix_lens', None)
            )
            snap.req_pool_indices = _safe_tolist(
                getattr(forward_batch, 'req_pool_indices', None)
            )

            positions = getattr(forward_batch, 'positions', None)
            if positions is not None and isinstance(positions, torch.Tensor):
                snap.positions = _safe_tolist(positions[:min(32, len(positions))])

            # Check invariants
            self._check_invariants(snap, forward_batch)

        except Exception as e:
            snap.anomalies.append(f"snapshot_error: {e}")

        if extra:
            for k, v in extra.items():
                if hasattr(snap, k):
                    setattr(snap, k, v)

        self._emit(snap)
        self._last_snapshot = snap
        return snap

    def log_draft_start(self, worker_name: str, batch_size: int, num_steps: int):
        """Log when a speculative draft phase begins."""
        if not ENABLED:
            return
        self._step_count += 1
        snap = PhaseSnapshot(
            phase=f"draft_start:{worker_name}",
            batch_size=batch_size,
        )
        snap.anomalies.append(f"step={self._step_count}, num_steps={num_steps}")
        self._emit(snap)

    def log_draft_end(
        self,
        worker_name: str,
        drafted_token_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ):
        """Log when a speculative draft phase ends."""
        if not ENABLED:
            return
        snap = PhaseSnapshot(phase=f"draft_end:{worker_name}")
        if drafted_token_ids is not None:
            snap.drafted_tokens = _safe_tolist(drafted_token_ids)
        if hidden_states is not None:
            snap.hidden_shape = str(list(hidden_states.shape))
            try:
                snap.hidden_norm = float(hidden_states.float().norm().item())
            except Exception:
                pass
        self._emit(snap)

    def log_verify(
        self,
        phase: str,
        accepted: Optional[int] = None,
        total_drafted: Optional[int] = None,
        forward_batch=None,
    ):
        """Log verify phase results."""
        if not ENABLED:
            return
        snap = PhaseSnapshot(phase=phase)
        snap.accepted_count = accepted
        if accepted is not None and total_drafted and total_drafted > 0:
            snap.acceptance_rate = accepted / total_drafted

        if forward_batch is not None:
            snap.seq_lens = _safe_tolist(getattr(forward_batch, 'seq_lens', None))
            snap.extend_seq_lens = _safe_tolist(
                getattr(forward_batch, 'extend_seq_lens', None)
            )
            snap.forward_mode = str(getattr(forward_batch, 'forward_mode', '?'))

        self._emit(snap)

    def log_attention_entry(
        self,
        backend_name: str,
        forward_batch,
        supplied_prefix: Optional[List] = None,
        recomputed_prefix: Optional[List] = None,
    ):
        """Log attention entry — especially if supplied vs recomputed prefix disagree."""
        if not ENABLED:
            return
        snap = PhaseSnapshot(phase=f"attention:{backend_name}")
        snap.forward_mode = str(getattr(forward_batch, 'forward_mode', '?'))
        snap.batch_size = getattr(forward_batch, 'batch_size', 0)
        snap.seq_lens = _safe_tolist(getattr(forward_batch, 'seq_lens', None))
        snap.extend_seq_lens = _safe_tolist(
            getattr(forward_batch, 'extend_seq_lens', None)
        )
        snap.extend_prefix_lens = _safe_tolist(
            getattr(forward_batch, 'extend_prefix_lens', None)
        )

        if supplied_prefix is not None and recomputed_prefix is not None:
            mismatches = []
            for i, (s, r) in enumerate(zip(supplied_prefix, recomputed_prefix)):
                if s != r:
                    mismatches.append(f"req[{i}]: supplied={s} recomputed={r}")
            if mismatches:
                snap.anomalies.append(
                    f"PREFIX_MISMATCH: {'; '.join(mismatches[:8])}"
                )
                if FAIL_FAST:
                    raise RuntimeError(
                        f"[SPEC_DIAG] prefix mismatch at attention entry: {mismatches}"
                    )

        self._emit(snap)

    def log_hidden_state_transition(
        self,
        phase: str,
        hidden_before: Optional[torch.Tensor],
        hidden_after: Optional[torch.Tensor],
        positions_before: Optional[torch.Tensor] = None,
        positions_after: Optional[torch.Tensor] = None,
    ):
        """Log hidden state changes across phase boundaries."""
        if not ENABLED:
            return
        snap = PhaseSnapshot(phase=f"hidden_transition:{phase}")

        if hidden_before is not None:
            snap.anomalies.append(
                f"h_before: {_tensor_stats(hidden_before)}"
            )
        if hidden_after is not None:
            snap.anomalies.append(
                f"h_after: {_tensor_stats(hidden_after)}"
            )
        if positions_before is not None:
            snap.anomalies.append(
                f"pos_before: {_safe_tolist(positions_before, 8)}"
            )
        if positions_after is not None:
            snap.anomalies.append(
                f"pos_after: {_safe_tolist(positions_after, 8)}"
            )

        self._emit(snap)

    @contextmanager
    def trace_phase(self, phase_name: str, **attrs):
        """Context manager that creates an OTEL span + logs entry/exit."""
        if not ENABLED:
            yield
            return

        tracer = _get_tracer()
        if tracer and _otel_available:
            from opentelemetry import trace as otrace
            with tracer.start_as_current_span(phase_name) as span:
                for k, v in attrs.items():
                    try:
                        span.set_attribute(f"spec.{k}", str(v))
                    except Exception:
                        pass
                start = time.time_ns()
                try:
                    yield
                except Exception as e:
                    span.set_attribute("spec.error", str(e))
                    span.set_status(
                        otrace.Status(otrace.StatusCode.ERROR, str(e))
                    )
                    raise
                finally:
                    dur_us = (time.time_ns() - start) / 1000
                    span.set_attribute("spec.duration_us", dur_us)
        else:
            start = time.time_ns()
            yield
            dur_us = (time.time_ns() - start) / 1000
            if VERBOSE:
                logger.info(
                    "[SPEC_DIAG] %s completed in %.0f µs", phase_name, dur_us
                )

    def _check_invariants(self, snap: PhaseSnapshot, fb):
        """Check critical invariants and log violations."""
        # Invariant 1: extend_seq_lens + extend_prefix_lens == seq_lens (for extend modes)
        mode_str = snap.forward_mode
        if "EXTEND" in mode_str or "VERIFY" in mode_str:
            seq = snap.seq_lens
            ext = snap.extend_seq_lens
            pfx = snap.extend_prefix_lens
            if seq and ext and pfx and len(seq) == len(ext) == len(pfx):
                for i in range(min(len(seq), 16)):
                    s = seq[i] if isinstance(seq[i], (int, float)) else 0
                    e = ext[i] if isinstance(ext[i], (int, float)) else 0
                    p = pfx[i] if isinstance(pfx[i], (int, float)) else 0
                    if abs((e + p) - s) > 1:
                        violation = (
                            f"INV_EXTEND_SUM: req[{i}] seq={s} != "
                            f"extend({e}) + prefix({p}) = {e+p}"
                        )
                        snap.anomalies.append(violation)
                        self._violations.append(violation)

        # Invariant 2: positions should be non-negative and < MAX_CONTEXT
        if snap.positions:
            for i, p in enumerate(snap.positions):
                if isinstance(p, (int, float)):
                    if p < 0:
                        v = f"INV_NEG_POS: positions[{i}]={p}"
                        snap.anomalies.append(v)
                        self._violations.append(v)
                    elif p > 65536:
                        v = f"INV_HUGE_POS: positions[{i}]={p}"
                        snap.anomalies.append(v)
                        self._violations.append(v)

        # Invariant 3: extend_seq_lens should be positive (non-zero) for extend modes
        if "EXTEND" in mode_str and snap.extend_seq_lens:
            for i, e in enumerate(snap.extend_seq_lens):
                if isinstance(e, (int, float)) and e <= 0:
                    v = f"INV_ZERO_EXTEND: extend_seq_lens[{i}]={e}"
                    snap.anomalies.append(v)
                    self._violations.append(v)

        if snap.anomalies and FAIL_FAST:
            raise RuntimeError(
                f"[SPEC_DIAG] invariant violations in {snap.phase}: "
                f"{snap.anomalies}"
            )

    def _emit(self, snap: PhaseSnapshot):
        """Emit snapshot as structured log + OTEL span attributes."""
        d = snap.to_dict()
        has_anomalies = bool(snap.anomalies)

        # Structured log
        if has_anomalies:
            logger.warning("[SPEC_DIAG] %s", json.dumps(d, default=str))
        elif VERBOSE:
            logger.info("[SPEC_DIAG] %s", json.dumps(d, default=str))

        # OTEL span event
        tracer = _get_tracer()
        if tracer and _otel_available:
            from opentelemetry import trace as otrace
            span = otrace.get_current_span()
            if span and span.is_recording():
                attrs = {}
                for k, v in d.items():
                    try:
                        attrs[f"spec.{k}"] = str(v)
                    except Exception:
                        pass
                span.add_event(snap.phase, attributes=attrs)

    def get_violations_summary(self) -> str:
        """Return summary of all invariant violations detected."""
        if not self._violations:
            return "No invariant violations detected."
        counts: Dict[str, int] = {}
        for v in self._violations:
            key = v.split(":")[0] if ":" in v else v
            counts[key] = counts.get(key, 0) + 1
        lines = [f"  {k}: {c} occurrences" for k, c in sorted(counts.items())]
        return f"Violations ({len(self._violations)} total):\n" + "\n".join(lines)


# Module-level singleton
_instance: Optional[SpecDecodeTracer] = None


def get_tracer() -> SpecDecodeTracer:
    global _instance
    if _instance is None:
        _instance = SpecDecodeTracer()
        if ENABLED:
            logger.info(
                "[SPEC_DIAG] Speculative decode tracer ENABLED "
                "(verbose=%s, fail_fast=%s)",
                VERBOSE, FAIL_FAST,
            )
    return _instance
