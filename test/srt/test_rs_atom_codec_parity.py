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


def test_policy_codec_for_matches_rust():
    """AutoQuantPolicy decision parity: one JSON loaded into both the Python policy class and
    the Rust PyAutoQuantPolicy must resolve codec_for(layer, stage) identically — pinning the
    *policy* half of the single-source-of-truth (not just the codec registry). Skipped (with a
    note) if sglang's heavy deps can't import in this env."""
    try:
        from sglang.srt.layers.quantization.autoquant.policy import (
            AutoQuantPolicy,
            CodecChoice,
            Stage,
        )
    except Exception as e:  # pragma: no cover - env-dependent
        print(f"[skip policy codec_for parity: {type(e).__name__}: {e}]")
        return

    py = AutoQuantPolicy(
        fingerprint_digest="digest123",
        model_family="qwen2",
        n_layers=4,
        layer_codecs={
            0: CodecChoice(codec="tq4", bit_width=4),
            1: CodecChoice(codec="rq4_planar", bit_width=4),
            2: CodecChoice(codec="rq3_planar", bit_width=3),
        },
        stage_overrides={Stage.DECODE: {1: CodecChoice(codec="tq2", bit_width=2)}},
    )
    blob = py.to_json()
    py_loaded = AutoQuantPolicy.from_json(blob)
    rust = rc.PyAutoQuantPolicy.from_json(blob)

    for layer in (0, 1, 2, 3):  # 3 is unmapped -> tq4 fallback on both sides
        for stage in (None, "prefill", "decode", "draft"):
            pc = py_loaded.codec_for(layer, Stage(stage) if stage else None)
            rcc = rust.codec_for(layer, stage)
            assert pc.codec == rcc["codec"], (layer, stage, pc.codec, rcc["codec"])
            assert pc.bit_width == rcc["bit_width"], (layer, stage)
            # Python CodecChoice.note defaults to ""; Rust serializes Some("") -> "" too.
            assert (pc.note or "") == (rcc["note"] or ""), (layer, stage, pc.note, rcc["note"])
    # spot-check the cross-impl values that matter most
    assert rust.codec_for(1, "decode")["codec"] == "tq2"     # stage override wins
    assert rust.codec_for(1, "prefill")["codec"] == "rq4_planar"  # falls through to layer codec
    assert rust.codec_for(3)["codec"] == "tq4"               # fallback agrees


def test_server_args_kv_cache_dtype_normalization_is_behavior_preserving():
    """Phase E tranche-3: ServerArgs._normalize_kv_cache_dtype routes Rust-known aliases
    through rs_atom_codec but must reproduce the OLD hand-rolled behavior EXACTLY for every
    accepted --kv-cache-dtype choice. Critically, bf16 and bfloat16 must stay DISTINCT
    (trtllm_mla validation accepts 'bf16', rejects 'bfloat16'), and torch.dtype objects must
    pass through untouched. Skipped (with note) if sglang can't import in this env."""
    try:
        from sglang.srt.server_args import ServerArgs
    except Exception as e:  # pragma: no cover - env-dependent
        print(f"[skip server_args normalization: {type(e).__name__}: {e}]")
        return
    from types import SimpleNamespace

    # input -> expected post-normalization (must equal the pre-tranche-3 behavior)
    expected = {
        "auto": "auto",
        "fp8_e5m2": "fp8_e5m2",
        "fp8_e4m3": "fp8_e4m3",
        "atom_fp8": "fp8_e4m3",          # Rust expansion == old map
        "rq3": "rq3_planar",             # Rust expansion == old _RQ_SHORTHAND
        "rq4": "rq4_planar",
        "rq3_planar": "rq3_planar",
        "rq4_planar": "rq4_planar",
        "rq3_iso": "rq3_iso",
        "rq4_iso": "rq4_iso",
        "tq2": "tq2",
        "tq3": "tq3",
        "tq4": "tq4",
        "bf16": "bf16",                  # MUST stay distinct (site server_args.py:2429)
        "bfloat16": "bfloat16",          # MUST NOT collapse to bf16
        "fp4_e2m1": "fp4_e2m1",          # unknown to Rust -> untouched
        "rq3_hybrid": "rq3_hybrid",      # engine-only composite -> untouched
        "univ_rq3": "univ_rq3",
        "kv_mixed": "kv_mixed4",         # retained _MIXED_SHORTHAND
        "kv_mixed3": "kv_mixed3",
    }
    for inp, exp in expected.items():
        ns = SimpleNamespace(kv_cache_dtype=inp)
        ServerArgs._normalize_kv_cache_dtype(ns)
        assert ns.kv_cache_dtype == exp, (inp, ns.kv_cache_dtype, exp)

    # torch.dtype objects pass through untouched (no AttributeError/TypeError crash)
    try:
        import torch

        for dt in (torch.float16, torch.bfloat16):
            ns = SimpleNamespace(kv_cache_dtype=dt)
            ServerArgs._normalize_kv_cache_dtype(ns)
            assert ns.kv_cache_dtype is dt, dt
    except ImportError:
        pass


if __name__ == "__main__":
    test_wheel_present()
    test_fallback_normalize_matches_rust()
    test_fallback_registry_matches_rust()
    test_fallback_bit_width_matches_rust()
    test_fallback_backend_plan_matches_rust()
    test_policy_valid_codecs_registry_derived_and_never_shrinks()
    test_policy_codec_for_matches_rust()
    test_server_args_kv_cache_dtype_normalization_is_behavior_preserving()
    print("ALL PARITY CHECKS PASS")
