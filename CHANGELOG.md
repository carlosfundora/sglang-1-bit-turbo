# Changelog

## [Unreleased]
### Added
- Added `atom` wiring surfaces for attention, MoE runner backend flag parsing, and FP8 GEMM backend flag parsing.
- Added `atom_fp8` KV cache dtype alias to normalize into `fp8_e4m3` during server argument post-processing.

### Changed
- Made the `cpp_ngram` draft-tree flatten deterministic (`Node.next` is now an ordered `std::map`), so the n-gram speculative draft-token order and attention-mask row indices no longer depend on hash-table iteration order — reproducible speculation run-to-run. (Synthesized from the deterministic `rs_ngram_draft` Rust port.)
- Registered a new `atom` attention backend path in `attention_registry` with explicit fallback to Triton when AITER is unavailable.
- Updated MoE and FP8 backend initializers to map `atom` to existing stable execution paths (`triton` for MoE, `aiter` for FP8 GEMM).
- Added server-args unit coverage for `atom`/`atom_fp8` CLI wiring.
- Optimized Triton attention backend kernel tuning (waves_per_eu, num_warps) for AMD RDNA2 (gfx1030) GPUs to prevent crashes and improve throughput.
- Optimized attention hot loops for RDNA2 hardware.
- Propagated RDNA2 default `mem_fraction_static` and `chunked_prefill_size` values in `server_args.py` when users do not set them explicitly.
- Optimized model loader startup by replacing `os.walk` with a fast Rust-based `find_files` utility using the `ignore` crate.
- Added a new `universal_broker` attention backend option that wraps Triton compute while recording hybrid universal KV broker metadata for `rq3_hybrid` and `univ_rq3` cache modes.
- Expanded `--kv-cache-dtype` choices with universal broker hybrid modes (`rq3_hybrid`, `univ_rq3`).
- Added `UNIVERSAL_KV` host-tier pool name and a pinned host allocation helper for warm-tier spill plumbing.
- Hardened `UniversalKVBroker` with GPU/RAM budget-aware hot-to-warm demotion, warm-tier eviction, and broker memory metrics for spill/eviction observability.

### Documentation
- Created: `.jules/reports/research/repo-triangulation-20260427-024844.md`
- Updated: `docs/rdna2_support.md`
- Added: `.jules/reports/research/repo-triangulation-20260426-120000.md`
- Updated: `.jules/journals/triangulator-forge.md`
- Updated: `.jules/journals/rocmancer.md`

### Added
- Support for Qwen 0.6B Embedding and Jina embedding models.
- Support for `int4_awq`, `w4a8_awq`, and `nvfp4_awq` in ModelOpt quantization configuration.
- Added new universal KV broker modules:
  - `python/sglang/srt/layers/attention/universal_kv_broker.py`
  - `python/sglang/srt/layers/attention/universal_broker_backend.py`
  - `python/sglang/srt/mem_cache/universal_kv_spill.py`
  - `universal_kv/types.py`, `universal_kv/model_registry.py`
- Added universal KV focused tests:
  - `test/srt/test_universal_kv_types.py`
  - `test/srt/test_universal_kv_broker.py`
  - `test/srt/test_universal_kv_server_args.py`
  - `test/srt/test_universal_kv_spill.py`
### Documentation
- Created: `.jules/reports/research/repo-triangulation-20260426-031204.md`
- Updated: `.jules/journals/triangulator-forge.md`
