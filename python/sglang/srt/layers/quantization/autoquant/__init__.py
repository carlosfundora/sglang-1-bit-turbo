"""AutoQuant — adaptive per-layer KV codec selection.

Discovers the best combination of {planar, iso, rq3, rq4, tq3, tq4}
per layer (and optionally per stage) for any model on any GPU, and
persists the policy to disk so future runs start hot.

Public API:
    from sglang.srt.layers.quantization.autoquant import (
        AutoQuantPolicy, load_policy, save_policy, detect_fingerprint,
    )

The runner (`MHATokenToKVPoolMixed`) consults `load_policy(fp)` at
init and either:
  - applies the per-layer codec map if a saved policy exists, OR
  - logs that no policy was found and falls back to the static
    `kv_auto` heuristic.

This module never blocks the inference hot path. All disk I/O is
init-time only; the policy is a small JSON shelf.
"""

from sglang.srt.layers.quantization.autoquant.policy import (
    AutoQuantPolicy,
    CodecChoice,
    Stage,
    POLICY_VERSION,
)
from sglang.srt.layers.quantization.autoquant.fingerprint import (
    AutoQuantFingerprint,
    detect_fingerprint,
)
from sglang.srt.layers.quantization.autoquant.shelf import (
    load_policy,
    save_policy,
    delete_policy,
    list_policies,
    cache_dir,
)

__all__ = [
    "AutoQuantPolicy",
    "CodecChoice",
    "Stage",
    "POLICY_VERSION",
    "AutoQuantFingerprint",
    "detect_fingerprint",
    "load_policy",
    "save_policy",
    "delete_policy",
    "list_policies",
    "cache_dir",
]
