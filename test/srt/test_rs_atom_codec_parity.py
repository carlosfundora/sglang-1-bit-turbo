# Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora · MIT
"""Drift guard for the ATOM-RS KV-codec single-source-of-truth (Phase E tranche-2).

Pins the in-Python fallback mirror in ``autoquant/_rs_codec.py`` to the authoritative Rust
binding ``rs_atom_codec`` so the two can never silently diverge (that divergence is the exact
problem the binding exists to kill). Also checks the ``_VALID_CODECS`` reconciliation in
``autoquant/policy.py`` is registry-derived and never shrinks the previously-accepted set.

Run standalone (uses the fork's sglang on PYTHONPATH and the venv's rs_atom_codec wheel):

    PYTHONPATH=python python test/srt/test_rs_atom_codec_parity.py

or under pytest. The fallback-vs-Rust parity does not import sglang; the policy check does and
is skipped (with a printed note) if sglang's heavy deps can't import in the current env.
"""

import importlib.util
import os

import rs_atom_codec as rc  # the authoritative Rust source of truth (must be installed)

_HERE = os.path.dirname(os.path.abspath(__file__))
_RS_CODEC_PATH = os.path.join(
    _HERE, "..", "..", "python", "sglang", "srt", "layers",
    "quantization", "autoquant", "_rs_codec.py",
)


def _load_shim():
    """Load _rs_codec.py standalone (it only needs `logging` + tries `rs_atom_codec`)."""
    spec = importlib.util.spec_from_file_location("_atom_rs_codec_shim", _RS_CODEC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wheel_present():
    assert rc.registry_codecs(), "rs_atom_codec wheel not installed / empty registry"


def test_fallback_normalize_matches_rust():
    shim = _load_shim()
    assert shim.HAVE_RS_ATOM_CODEC, "shim did not pick up the Rust binding"
    for alias, expected in shim._ALIAS_TO_CANONICAL.items():
        assert shim._fallback_normalize(alias) == rc.normalize_codec_alias(alias) == expected, alias
    # unknown alias parity
    for bad in ("planar", "iso", "kv_mixed", "fp16", "not_a_codec"):
        rust_ok = rc.is_known_codec(bad)
        fb_ok = bad.lower() in shim._ALIAS_TO_CANONICAL
        assert rust_ok == fb_ok, bad


def test_fallback_registry_matches_rust():
    shim = _load_shim()
    assert set(shim._REGISTRY_CODECS) == set(rc.registry_codecs()), (
        sorted(set(shim._REGISTRY_CODECS) ^ set(rc.registry_codecs()))
    )


def test_fallback_bit_width_matches_rust():
    shim = _load_shim()
    for alias in shim._ALIAS_TO_CANONICAL:
        assert shim._BIT_WIDTH[shim._fallback_normalize(alias)] == rc.codec_bit_width(alias), alias


def test_fallback_backend_plan_matches_rust():
    shim = _load_shim()
    probe = set(rc.registry_codecs()) | {"bf16", "auto", "int8", "fp8_e5m2", "rq3", "atom_fp8"}
    for alias in sorted(probe):
        assert shim._fallback_backend_plan(alias) == rc.backend_plan_for(alias), alias


def test_policy_valid_codecs_registry_derived_and_never_shrinks():
    """Skipped (with note) if sglang's heavy deps can't import in this env."""
    try:
        from sglang.srt.layers.quantization.autoquant import policy as _policy
    except Exception as e:  # pragma: no cover - env-dependent
        print(f"[skip policy import check: {type(e).__name__}: {e}]")
        return
    valid = _policy._VALID_CODECS
    # registry-derived: every authoritative codec is valid
    assert set(rc.registry_codecs()) <= valid, sorted(set(rc.registry_codecs()) - valid)
    # never-shrink: every codec the runner accepted before is still valid
    old_accepted = {
        "rq3", "rq4", "rq4_planar", "rq3_planar", "tq3", "tq4",
        "planar", "iso", "kv_mixed", "kv_mixed3", "kv_mixed4", "fp16", "bf16",
    }
    assert old_accepted <= valid, sorted(old_accepted - valid)
    # newly-adopted Rust codecs that the old hand-rolled set was missing
    for extra in ("tq1", "tq2", "tq8", "rq3_iso", "rq4_iso", "fp8_e4m3"):
        assert extra in valid, extra


if __name__ == "__main__":
    test_wheel_present()
    test_fallback_normalize_matches_rust()
    test_fallback_registry_matches_rust()
    test_fallback_bit_width_matches_rust()
    test_fallback_backend_plan_matches_rust()
    test_policy_valid_codecs_registry_derived_and_never_shrinks()
    print("ALL PARITY CHECKS PASS")
