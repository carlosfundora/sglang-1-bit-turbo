"""AutoQuant on-disk policy shelf.

Default location: ``~/.cache/sglang/autoquant/<digest>.json``
Override via env var ``SGLANG_AUTOQUANT_DIR``.

Fingerprint-keyed; never returns a stale policy from a different
runtime environment. All operations are best-effort + logged — this
module never raises into the inference hot path.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from sglang.srt.layers.quantization.autoquant.fingerprint import AutoQuantFingerprint
from sglang.srt.layers.quantization.autoquant.policy import AutoQuantPolicy

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "~/.cache/sglang/autoquant"


def cache_dir() -> Path:
    """Resolve the active AutoQuant cache directory and ensure it exists."""
    raw = os.environ.get("SGLANG_AUTOQUANT_DIR", _DEFAULT_CACHE_DIR)
    p = Path(os.path.expanduser(raw))
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("AutoQuant: could not create cache dir %s: %s", p, e)
    return p


def _policy_path(fp: AutoQuantFingerprint) -> Path:
    return cache_dir() / f"{fp.hex_digest}.json"


def load_policy(fp: AutoQuantFingerprint) -> Optional[AutoQuantPolicy]:
    """Return the saved policy for `fp`, or None if no cache hit / on error."""
    path = _policy_path(fp)
    if not path.exists():
        logger.info("AutoQuant: no saved policy for fingerprint %s", fp.hex_digest)
        return None
    try:
        text = path.read_text()
        policy = AutoQuantPolicy.from_json(text)
        if policy.fingerprint_digest != fp.hex_digest:
            logger.warning(
                "AutoQuant: policy at %s has digest %s, expected %s — ignoring",
                path, policy.fingerprint_digest, fp.hex_digest,
            )
            return None
        logger.info(
            "AutoQuant: loaded policy %s (learner=%s, layers=%d, hist=%s)",
            fp.hex_digest, policy.learner, policy.n_layers,
            policy.codec_histogram(),
        )
        return policy
    except Exception as e:
        logger.warning("AutoQuant: failed to load policy %s: %s", path, e)
        return None


def save_policy(policy: AutoQuantPolicy) -> Optional[Path]:
    """Atomically write `policy` to disk, keyed by its fingerprint digest.

    Returns the written path, or None on error.
    """
    if not policy.created_at:
        # Stamp creation time on first save.
        policy.created_at = time.time()
    target = cache_dir() / f"{policy.fingerprint_digest}.json"
    try:
        # Atomic write: tmp + rename. Survives partial-write crashes.
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            f.write(policy.to_json())
            tmp = Path(f.name)
        os.replace(str(tmp), str(target))
        logger.info(
            "AutoQuant: saved policy %s (learner=%s, score=%s) → %s",
            policy.fingerprint_digest, policy.learner, policy.score, target,
        )
        return target
    except Exception as e:
        logger.warning("AutoQuant: failed to save policy %s: %s", target, e)
        return None


def delete_policy(fp: AutoQuantFingerprint) -> bool:
    """Remove the cached policy for `fp`. Returns True if a file was removed."""
    path = _policy_path(fp)
    if not path.exists():
        return False
    try:
        path.unlink()
        logger.info("AutoQuant: deleted policy %s", path)
        return True
    except Exception as e:
        logger.warning("AutoQuant: failed to delete policy %s: %s", path, e)
        return False


def list_policies() -> List[Path]:
    """Return all known policy files in the cache."""
    d = cache_dir()
    try:
        return sorted(d.glob("*.json"))
    except Exception as e:
        logger.warning("AutoQuant: failed to list cache dir %s: %s", d, e)
        return []
