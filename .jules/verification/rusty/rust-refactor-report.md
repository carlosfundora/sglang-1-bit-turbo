# Rusty Rust Refactor Report

## Repository Recon
Analyzed `python/sglang/srt/parser/jinja_template_utils.py`, `python/sglang/srt/mem_cache/radix_cache.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py`, and utilities across `python/sglang/utils.py`. The MambaRadixCache was skipped due to its heavy Python memory-pool and locking FFI constraints which make it highly complex for a one-shot run. Jinja template logic inside `detect_jinja_template_content_format` relies on deep AST traversal which explicitly violates pure-regex constraints in memory guidelines.

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `find_printable_text` (`utils.py`) | Python | Lower string/unicode-slicing overhead on stream | Low | Low | Selected |
| 2 | `detect_jinja_template_content_format` | Python | Avoid Jinja AST overhead | High | High | Rejected |
| 3 | `MambaRadixCache` (`_insert_helper`) | Python | Faster cache operations | Very High | High | Rejected |
| 4 | `rpd_to_chrome_trace` | Python | Faster sqlite conversion offline | Medium | Medium | Rejected |

## Selected Candidate

- **Path**: `python/sglang/utils.py` and `python/sglang/rust_utils/src/lib.rs`
- **Current implementation**: Python function utilizing `.endswith()`, `ord()`, and string slicing backwards logic checking CJK boundaries.
- **Rust replacement**: Native `find_printable_text` via PyO3, which replicates exact byte and char boundaries correctly and avoids repeated python function frames in hot loops.
- **Reason selected**: Fits perfectly within the constrained run boundaries. Safe to port without cascading memory pool dependencies. Benchmark results demonstrate clear latency reduction for the token streaming path.

## Implementation Summary
Added `is_chinese_char` helper function returning standard boolean matching Python's boundary checks. Added `find_printable_text` to the `sglang_rust_utils` module which explicitly utilizes UTF-8 `char_indices` iterators to reverse string offsets while mimicking Python slicing semantics properly. Graceful fallback logic remains if the Rust PyO3 library is absent.

## Before Benchmark
`duration_ms`: 3426.8ms (1,000,000 runs)

## After Benchmark
`duration_ms`: 2151.9ms (1,000,000 runs)

## Benchmark Delta
Approx. **37.2% latency reduction**

## Tests Run
Run `test_find_printable_text.py` parity tests on normal ASCII, CJK tokens, mixed strings, whitespace boundaries, and various CJK-extension plane symbols. Tests **PASSED**.

## Files Changed
- `python/sglang/rust_utils/src/lib.rs`
- `python/sglang/utils.py`
- `test/registered/unit/test_find_printable_text.py`

## Compatibility Notes
Ensured strict utf8 character boundary validation in Rust match the unicode representations expected by Python strings via length offsets tracking.

## Remaining Follow-Ups
MambaRadixCache would be a good future task, but likely requires porting the custom `memory_pool` allocators to C++/Rust first to avoid heavy PyO3 dictionary mappings.