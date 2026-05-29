# Rusty Rust Refactor Report

## Repository Recon
The codebase uses a mixture of Python and Rust (via PyO3) for high-performance sections. It contains `sglang_rust_utils` which exposes various functionality like JSON Schema parsing and prefix hashing. During my recon, I found that some of the candidates listed in `rust_candidates.md` like `ConfigArgumentMerger` and parts of `jinja_template_utils.py` and `reasoning_parser.py` already had Rust alternatives implemented or were found to be slower through FFI than Python string manipulation.

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `extract_longbench_v2_answer` (`python/sglang/test/simple_eval_longbench_v2.py`) | Python | Faster repeated regex evaluations | Low | Low | Selected |
| 2 | `saguaro_prefix_hash` | Python/Rust | Faster string hashing | Low | Low | Already Implemented |
| 3 | `detect_jinja_template_content_format` | Python/Rust | Faster AST inspection | High | Medium | Already Implemented |
| 4 | `ReasoningParser` detectors | Python/Rust | Reduced text loop overhead | Medium | Medium | Evaluated, Slower via FFI |
| 5 | `murmur_hash32` | Triton | GPU Kernel | High | High | Ignored |

## Selected Candidate

- Path: `python/sglang/test/simple_eval_longbench_v2.py`
- Current implementation: Pure Python regex execution.
- Rust replacement: Pre-compiled `regex::Regex` via `OnceLock` running through PyO3 FFI.
- Reason selected: Clean, bounded change, easily verifiable and directly impactful in evaluation loop processing. Provides a ~2x performance speedup.

## Implementation Summary
Created a new Rust module `longbench.rs` within `sglang_rust_utils` and exported `extract_longbench_v2_answer` via the existing PyO3 module interface. Implemented the 4 fallback Regex conditions with `OnceLock` for performance, mapping the Python behavior exactly. Handled FFI bindings and fallback Python logic gracefully.

## Before Benchmark
- **Duration:** 211.11 ms
- **Iterations:** 100000

## After Benchmark
- **Duration:** 112.44 ms
- **Iterations:** 100000

## Benchmark Delta
- **Change:** -46.74%
- The Rust implementation is approximately twice as fast.

## Tests Run
Custom standalone test runner executed simulating the evaluation runner imports because of mock failures in the sandbox environment.

## Files Changed
- `python/sglang/test/simple_eval_longbench_v2.py`
- `python/sglang/rust_utils/src/lib.rs`
- `python/sglang/rust_utils/src/longbench.rs`

## Compatibility Notes
Fallback to Python regex logic remains intact if the Rust module is unable to be imported.

## Remaining Follow-Ups
Ensure that `sglang_rust_utils` is built and distributed with releases properly so the CI takes advantage of this optimization.
