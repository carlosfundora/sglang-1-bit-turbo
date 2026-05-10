# Rusty Rust Refactor Report

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `prefix_hold` in `harmony_parser.py` | Python | High (Hot path, string iterations) | Low | Low | Selected |
| 2 | `log_parser.py` | Python | Medium | Medium | Low | Rejected |
| 3 | `ConfigArgumentMerger` | Python/Rust | N/A (Already optimized in Rust) | N/A | N/A | Rejected |

## Selected Candidate

- Path: `python/sglang/srt/parser/harmony_parser.py`
- Current implementation: Pure python string iteration and slicing for detecting text prefixes in chunks during generation.
- Rust replacement: `prefix_hold` implemented in `sgl-model-gateway/bindings/python/src/harmony_parser.rs` and bound via PyO3 under `sglang_router_rs`.
- Reason selected: Clean input/output boundary, executes synchronously during the hot streaming path, avoids large rewrites or complex architectures.

## Implementation Summary
Created `sgl-model-gateway/bindings/python/src/harmony_parser.rs` implementing `prefix_hold` with utf-8 boundary aware checking logic to ensure memory safety. Included module in `lib.rs`.
Modified `python/sglang/srt/parser/harmony_parser.py` to `try: from sglang_router.sglang_router_rs import prefix_hold`, utilizing python implementation as a graceful fallback.

## Before Benchmark
{"duration_ms": 1028.77, "throughput": "97202.8 ops/sec"}

## After Benchmark
{"duration_ms": 964.29, "throughput": "103702.8 ops/sec"}

## Benchmark Delta
Real benchmark time is ~0.96s per 100k iterations in Rust, reducing Python's ~1.03s, yielding roughly ~7% reduction while ensuring full multi-byte character boundary safety over the stream.

## Tests Run
Parity test run `test_harmony2.py` ensures output equality for the reasoning parser fallback mechanism across ascii and unicode string combinations. Passed.
Linter test passed via `ruff`.

## Files Changed
- `sgl-model-gateway/bindings/python/src/lib.rs`
- `sgl-model-gateway/bindings/python/src/harmony_parser.rs`
- `python/sglang/srt/parser/harmony_parser.py`

## Compatibility Notes
Fallback `prefix_hold` retained for platforms/environments unable to compile the `sglang_router_rs` extension.

## Remaining Follow-Ups
None.
