# Changelog

## [Unreleased]
- Added robust, unified error handling definitions (`GfxGraphError`) and error reporting (`report_error`).
- Implemented conditional compilation integration (via optional `"logly"` feature) with `rs_logly_logger`.
- Provided a zero-cost system-level stderr fallback for non-logly environments.
- Updated documentation and README with comprehensive architecture and usage guidelines.
- Added the canonical `rs_gfxgraph_core` crate with schema, stats, and adapter modules.

