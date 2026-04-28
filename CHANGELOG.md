
## [Unreleased]

### Changed
- Optimized Triton attention backend kernel tuning (waves_per_eu, num_warps) for AMD RDNA2 (gfx1030) GPUs to prevent crashes and improve throughput.
### Documentation
- Added: `.jules/reports/research/repo-triangulation-20260426-120000.md`
- Updated: `.jules/journals/triangulator-forge.md`
- None required.

### Added
- Support for Qwen 0.6B Embedding and Jina embedding models.
- Support for `int4_awq`, `w4a8_awq`, and `nvfp4_awq` in ModelOpt quantization configuration.
### Documentation
- Created: `.jules/reports/research/repo-triangulation-20260426-031204.md`
- Updated: `.jules/journals/triangulator-forge.md`
