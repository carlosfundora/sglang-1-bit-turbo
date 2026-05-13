# Changelog

## [Unreleased]
### Documentation
- Created: `.jules/reports/research/repo-triangulation-20260427-024844.md`
- Updated: `docs/rdna2_support.md`

### Updated
- Optimized attention hot loops for RDNA2 hardware.
- Propagated RDNA2 default `mem_fraction_static` and `chunked_prefill_size` values in `server_args.py` when users do not set them explicitly.
- Added: `.jules/reports/research/repo-triangulation-20260426-120000.md`
- Updated: `.jules/journals/triangulator-forge.md`
- Updated: `.jules/journals/triangulator-forge.md`
- Optimized model loader startup by replacing `os.walk` with a fast Rust-based `find_files` utility using the `ignore` crate.

### Added
- Support for Qwen 0.6B Embedding and Jina embedding models.
- Support for `int4_awq`, `w4a8_awq`, and `nvfp4_awq` in ModelOpt quantization configuration.
- Created: `.jules/reports/research/repo-triangulation-20260426-031204.md`
- Updated: `.jules/journals/triangulator-forge.md`
