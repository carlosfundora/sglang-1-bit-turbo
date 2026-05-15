# Changelog

## [Unreleased]
### Added
- Added `atom` wiring surfaces for attention, MoE runner backend flag parsing, and FP8 GEMM backend flag parsing.
- Added `atom_fp8` KV cache dtype alias to normalize into `fp8_e4m3` during server argument post-processing.

### Updated
- Registered a new `atom` attention backend path in `attention_registry` with explicit fallback to Triton when AITER is unavailable.
- Updated MoE and FP8 backend initializers to map `atom` to existing stable execution paths (`triton` for MoE, `aiter` for FP8 GEMM).
- Added server-args unit coverage for `atom`/`atom_fp8` CLI wiring.

### Documentation
- Created: `.jules/reports/research/repo-triangulation-20260427-024844.md`
- Updated: `docs/rdna2_support.md`

### Updated
- Optimized attention hot loops for RDNA2 hardware.
- Propagated RDNA2 default `mem_fraction_static` and `chunked_prefill_size` values in `server_args.py` when users do not set them explicitly.
- Added: `.jules/reports/research/repo-triangulation-20260426-120000.md`
- Updated: `.jules/journals/triangulator-forge.md`
- Updated: `.jules/journals/triangulator-forge.md`

### Added
- Support for Qwen 0.6B Embedding and Jina embedding models.
- Support for `int4_awq`, `w4a8_awq`, and `nvfp4_awq` in ModelOpt quantization configuration.
- Created: `.jules/reports/research/repo-triangulation-20260426-031204.md`
- Updated: `.jules/journals/triangulator-forge.md`
