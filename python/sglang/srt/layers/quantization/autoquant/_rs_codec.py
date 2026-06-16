# Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora · MIT
"""Single entry point for ATOM-RS KV-codec resolution in sglang (Phase E tranche-2).

Prefers the authoritative Rust binding ``rs_atom_codec`` — the single source of truth shared
with the ATOM-RS engine (``rs_kv_quant_contracts`` + ``rs_kv_codec_adapters`` +
``rs_autoquant_policy``). If the wheel is not installed it falls back to a thin Python mirror and
logs a warning. The mirror is **pinned to the Rust output** by
``test/srt/test_rs_atom_codec_parity.py`` so the two cannot silently drift (that drift is exactly
the divergence this module exists to kill).

Codec *execution* stays in the engine kernels (Triton/HIP). This module only resolves the
*decision*: alias → canonical name, the authoritative valid-codec set, the codec→backend plan,
and bit width.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import rs_atom_codec as _rs

    HAVE_RS_ATOM_CODEC = True
except Exception as _e:  # pragma: no cover - exercised only when the wheel is absent
    _rs = None
    HAVE_RS_ATOM_CODEC = False
    logger.warning(
        "rs_atom_codec (authoritative Rust KV-codec binding) not importable (%s). "
        "Falling back to the in-Python mirror; build/install it from "
        "projects/rust/bindings/rs_atom_codec_pyo3 to use the single source of truth.",
        _e,
    )

# --- Python fallback mirror (pinned to the Rust output by the parity test) -------------------
# Mirrors rs_kv_quant_contracts::normalize_codec_alias.
_ALIAS_TO_CANONICAL = {
    "auto": "auto",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp8_e4m3": "fp8_e4m3",
    "atom_fp8": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
    "int8": "int8",
    "tq1": "tq1",
    "tq2": "tq2",
    "tq3": "tq3",
    "tq4": "tq4",
    "tq8": "tq8",
    "rq3": "rq3_planar",
    "rq3_planar": "rq3_planar",
    "rq4": "rq4_planar",
    "rq4_planar": "rq4_planar",
    "rq3_iso": "rq3_iso",
    "rq4_iso": "rq4_iso",
}
# Mirrors CodecAdapterRegistry::baseline() — the codecs that have a backend plan.
_REGISTRY_CODECS = (
    "tq1", "tq2", "tq3", "tq4", "tq8",
    "rq3_planar", "rq4_planar", "rq3_iso", "rq4_iso",
    "fp8_e4m3", "int8",
)
_BIT_WIDTH = {
    "auto": None, "bf16": None,
    "fp8_e4m3": 8, "fp8_e5m2": 8, "int8": 8,
    "tq1": 1, "tq2": 2, "tq3": 3, "tq4": 4, "tq8": 8,
    "rq3_planar": 3, "rq4_planar": 4, "rq3_iso": 3, "rq4_iso": 4,
}
_FAMILY = {
    "tq1": "turbo", "tq2": "turbo", "tq3": "turbo", "tq4": "turbo", "tq8": "turbo",
    "rq3_planar": "rotor_planar", "rq4_planar": "rotor_planar",
    "rq3_iso": "rotor_iso", "rq4_iso": "rotor_iso",
    "fp8_e4m3": "fp8", "int8": "int8",
}
_PREFERRED_BACKEND = {
    "turbo": "turboquant", "rotor_planar": "rotorquant",
    "rotor_iso": "rotorquant", "fp8": "fp8", "int8": "int8",
}


def _fallback_normalize(alias: str) -> str:
    key = alias.lower()
    try:
        return _ALIAS_TO_CANONICAL[key]
    except KeyError:
        raise ValueError(f"unsupported kv codec alias: {key}")


def _fallback_backend_plan(alias: str):
    codec = _fallback_normalize(alias)
    family = _FAMILY.get(codec)
    if family is None:  # auto/bf16/fp8_e5m2/int8 have no baseline registry plan
        return None
    pref = _PREFERRED_BACKEND[family]
    return {
        "codec": codec,
        "family": family,
        "preferred_backend": pref,
        "fallback_backend": "triton",
        "ultimate_fallback": "fp16",
        "backend_chain": [pref, "triton", "fp16"],
        "supported": True,
        "bit_width": _BIT_WIDTH[codec],
        "is_experimental": codec == "tq1",
    }


# --- Public API (Rust-first, fallback otherwise) --------------------------------------------
def normalize_codec_alias(alias: str) -> str:
    """Canonical snake_case codec name (e.g. ``rq3`` -> ``rq3_planar``). Raises ValueError."""
    return _rs.normalize_codec_alias(alias) if HAVE_RS_ATOM_CODEC else _fallback_normalize(alias)


def is_known_codec(alias: str) -> bool:
    """True if ``alias`` resolves to a known ATOM-RS codec (never raises)."""
    if HAVE_RS_ATOM_CODEC:
        return _rs.is_known_codec(alias)
    return alias.lower() in _ALIAS_TO_CANONICAL


def codec_bit_width(alias: str):
    """Canonical bit width (None for auto/bf16). Raises ValueError if unknown."""
    if HAVE_RS_ATOM_CODEC:
        return _rs.codec_bit_width(alias)
    return _BIT_WIDTH[_fallback_normalize(alias)]


def registry_codecs() -> list:
    """The authoritative valid-codec set (canonical names) — source of truth for _VALID_CODECS."""
    return list(_rs.registry_codecs()) if HAVE_RS_ATOM_CODEC else list(_REGISTRY_CODECS)


def backend_plan_for(alias: str):
    """Codec -> backend dispatch plan dict (or None if no registry plan). Raises if unknown."""
    return _rs.backend_plan_for(alias) if HAVE_RS_ATOM_CODEC else _fallback_backend_plan(alias)
