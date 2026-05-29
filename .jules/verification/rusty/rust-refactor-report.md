# Rusty Rust Refactor Report

## Blocker Summary: Task Superseded & Architecture Constraints

**Blocker Reason:** All high-value pure-Rust refactor candidates identified in `rust_candidates.md` have either already been implemented, or their refactoring would result in architectural regression/performance degradation due to FFI overhead.

### Files Inspected
1. `python/sglang/srt/parser/jinja_template_utils.py`
2. `python/sglang/srt/layers/utils/hash.py`
3. `python/sglang/srt/parser/reasoning_parser.py`
4. `python/sglang/srt/server_args.py`
5. `python/sglang/rust_utils/src/lib.rs`
6. `python/sglang/srt/mem_cache/mamba_radix_cache.py`
7. `python/sglang/srt/function_call/utils.py`
8. `python/sglang/srt/mem_cache/cpp_radix_tree/radix_tree.py`

### Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `ReasoningParser` | Python | CPU overhead reduction | Medium | Low | **Rejected** - Already refactored to Rust via `RustReasoningState`. |
| 2 | `ConfigArgumentMerger` | Python | Startup time | Low | Low | **Rejected** - Already refactored to Rust. |
| 3 | `jinja_template_utils.py` | Python | Faster prompt parsing | High | Medium | **Rejected** - Already refactored using `rust_detect` and `rust_process`. |
| 4 | `trim_overlap` | Python | Faster cache alignment | Low | Low | **Rejected** - Already implemented in `rust_utils`. |
| 5 | `murmur_hash32` | Triton | Faster hashing | Low | High | **Rejected** - Code is designed for GPU compilation via Triton JIT. |
| 6 | `mamba_radix_cache.py` | Python | Faster graph traversal | High | High | **Rejected** - Radix cache operations are already heavily ported to C++ bindings in `cpp_radix_tree`. |
| 7 | JSON Schema `infer_type_from_json_schema` | Python | Fast tool call extraction | Medium | Low | **Rejected** - Tried implementing but PyO3 FFI boundary cost on recursive dictionaries outweighs the logic speedup. |

## Implementation Summary

Attempted to port `infer_type_from_json_schema` and `_get_tool_schema_defs` from `python/sglang/srt/function_call/utils.py` into pure Rust. The pure Python version utilizes standard dictionary and string comparison loops which map to optimized underlying C code in Python. Moving complex recursive dictionaries back and forth across the PyO3 FFI boundary significantly degrades performance. No new high-impact targets exist without breaking architecture principles or replacing C++ libraries.

## Remaining Follow-Ups

Future efforts should identify candidates that operate on contiguous arrays or long plain-text payloads, rather than deeply nested structures like dictionaries or small AST fragments, where PyO3 serialization costs offset execution gains.
