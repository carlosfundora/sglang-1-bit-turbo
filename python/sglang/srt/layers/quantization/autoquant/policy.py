"""AutoQuant policy — per-layer (and optionally per-stage) codec choices."""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

POLICY_VERSION = 1


class Stage(str, Enum):
    """Inference stage at which a codec choice may differ."""

    PREFILL = "prefill"
    DECODE = "decode"
    DRAFT = "draft"  # speculative draft model


# Set of codec names recognised by the runner. These map 1:1 to the
# kv_cache_dtype strings that MHATokenToKVPool* already accepts.
_VALID_CODECS = frozenset(
    {
        # rotor family
        "rq3", "rq4", "rq4_planar", "rq3_planar",
        # turbo family
        "tq3", "tq4",
        # planar / iso scalar quant (used inside rotor too)
        "planar", "iso",
        # mixed K=TQ V=RQ
        "kv_mixed", "kv_mixed3", "kv_mixed4",
        # passthrough
        "fp16", "bf16",
    }
)


@dataclass(frozen=True)
class CodecChoice:
    """One codec choice for one layer (and optionally one stage)."""

    codec: str            # one of _VALID_CODECS
    bit_width: int = 4    # 3 or 4
    note: str = ""        # human-readable rationale ("low kurtosis K", etc.)

    def __post_init__(self):
        if self.codec not in _VALID_CODECS:
            # Don't hard-fail at construction — older policy files may
            # carry codec names we haven't registered yet. Log + accept.
            logger.warning(
                "AutoQuant: unknown codec %r in CodecChoice (allowed: %s). "
                "Falling back to runtime validation.",
                self.codec, sorted(_VALID_CODECS),
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CodecChoice":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AutoQuantPolicy:
    """A learned per-layer (and optionally per-stage) KV codec map.

    `layer_codecs[i]` is the default choice for layer `i`.
    `stage_overrides[stage][i]` may override that for a specific stage.
    """

    fingerprint_digest: str
    model_family: str
    n_layers: int
    layer_codecs: Dict[int, CodecChoice] = field(default_factory=dict)
    stage_overrides: Dict[Stage, Dict[int, CodecChoice]] = field(default_factory=dict)
    # Bookkeeping
    version: int = POLICY_VERSION
    learner: str = "manual"          # "manual" | "calibration_sweep" | "bandit"
    created_at: float = 0.0
    score: Optional[float] = None    # composite calibration score, if any

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def codec_for(self, layer_idx: int, stage: Optional[Stage] = None) -> CodecChoice:
        """Resolve the codec for a given layer + stage, with safe fallback."""
        if stage is not None and stage in self.stage_overrides:
            choice = self.stage_overrides[stage].get(layer_idx)
            if choice is not None:
                return choice
        choice = self.layer_codecs.get(layer_idx)
        if choice is not None:
            return choice
        # Last-resort: TQ4 on gfx1030 is the safe default per kv_auto.
        return CodecChoice(codec="tq4", bit_width=4, note="autoquant fallback")

    def is_uniform(self) -> bool:
        """True if every layer has the same codec choice (no per-layer win yet)."""
        if not self.layer_codecs:
            return True
        first = next(iter(self.layer_codecs.values()))
        return all(
            c.codec == first.codec and c.bit_width == first.bit_width
            for c in self.layer_codecs.values()
        )

    def codec_histogram(self) -> Dict[str, int]:
        """Quick `{"tq4": 18, "rq4_planar": 6, ...}` summary for logging."""
        hist: Dict[str, int] = {}
        for c in self.layer_codecs.values():
            key = f"{c.codec}:{c.bit_width}"
            hist[key] = hist.get(key, 0) + 1
        return hist

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_json(self) -> str:
        d = {
            "version": self.version,
            "fingerprint_digest": self.fingerprint_digest,
            "model_family": self.model_family,
            "n_layers": self.n_layers,
            "learner": self.learner,
            "created_at": self.created_at,
            "score": self.score,
            "layer_codecs": {
                str(k): v.to_dict() for k, v in self.layer_codecs.items()
            },
            "stage_overrides": {
                stage.value: {str(k): v.to_dict() for k, v in m.items()}
                for stage, m in self.stage_overrides.items()
            },
        }
        return json.dumps(d, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "AutoQuantPolicy":
        d = json.loads(s)
        ver = d.get("version", 1)
        if ver != POLICY_VERSION:
            logger.warning(
                "AutoQuant: policy version mismatch (file=%d, code=%d). "
                "Loading anyway; behaviour may differ.",
                ver, POLICY_VERSION,
            )
        layer_codecs = {
            int(k): CodecChoice.from_dict(v)
            for k, v in d.get("layer_codecs", {}).items()
        }
        stage_overrides: Dict[Stage, Dict[int, CodecChoice]] = {}
        for stage_name, m in d.get("stage_overrides", {}).items():
            try:
                stage = Stage(stage_name)
            except ValueError:
                logger.warning("AutoQuant: unknown stage %r in policy", stage_name)
                continue
            stage_overrides[stage] = {
                int(k): CodecChoice.from_dict(v) for k, v in m.items()
            }
        return cls(
            fingerprint_digest=d["fingerprint_digest"],
            model_family=d["model_family"],
            n_layers=d["n_layers"],
            layer_codecs=layer_codecs,
            stage_overrides=stage_overrides,
            version=ver,
            learner=d.get("learner", "manual"),
            created_at=d.get("created_at", 0.0),
            score=d.get("score"),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def uniform(
        cls,
        fingerprint_digest: str,
        model_family: str,
        n_layers: int,
        codec: str,
        bit_width: int = 4,
        note: str = "",
    ) -> "AutoQuantPolicy":
        """Build a uniform policy — useful as a baseline / kv_auto upgrade."""
        choice = CodecChoice(codec=codec, bit_width=bit_width, note=note)
        return cls(
            fingerprint_digest=fingerprint_digest,
            model_family=model_family,
            n_layers=n_layers,
            layer_codecs={i: choice for i in range(n_layers)},
            learner="uniform",
        )
