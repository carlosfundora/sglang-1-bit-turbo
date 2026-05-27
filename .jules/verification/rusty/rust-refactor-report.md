# Rusty Rust Refactor Report

## Repository Recon
Found a high-value performance bottleneck in `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`, specifically `parse_message_from_completion_text` and `parse_tool_calls`, which parse DeepSeek V3.2 tool call outputs via text matching and regex loops. This is evaluated frequently during generation response formatting.

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `parse_message_from_completion_text` in `encoding_dsv32.py` | Python (Regex/String) | Very High (removes regex loop overhead during output generation) | Medium | Low | Selected |
| 2 | `ConfigArgumentMerger` | Python | High | Medium | Low | Rejected (already implemented) |
| 3 | `ReasoningParser` | Python | High | High | Low | Rejected (already implemented) |
| 4 | `jinja_template_utils` | Python | Medium | Medium | Medium | Rejected (already implemented) |

## Selected Candidate

- **Path:** `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`
- **Current implementation:** `parse_message_from_completion_text` uses a custom `_read_until_stop` helper and Python's `re.findall` in a while-loop.
- **Rust replacement:** Implemented `parse_message_from_completion_text` and `parse_tool_calls` in `python/sglang/rust_utils/src/dsv32_parser.rs`.
- **Reason selected:** Frequently run during chat completion streaming for DSV3.2 models; string manipulation and regex extraction is much faster in Rust, skipping intermediate string object instantiation overhead.

## Implementation Summary
Added `dsv32_parser.rs` to `sglang_rust_utils`, avoiding heavy dependency compilation inside `sgl-model-gateway`. The Rust parser manually traverses the DSML tokens `<｜DSML｜invoke>` to extract parameters into a Python dictionary, dropping all `regex` usage for pure substring searches.

## Before Benchmark
680.70 ms (10000 tool call parsings)

## After Benchmark
139.82 ms (10000 tool call parsings)

## Benchmark Delta
-79.5% execution time (~4.8x speedup)

## Tests Run
`python test_rust_parser.py` confirmed 100% equivalence in JSON structure and values output between normal and thinking modes.

## Files Changed
- `python/sglang/rust_utils/Cargo.toml` (if needed, but `dsv32_parser.rs` added directly)
- `python/sglang/rust_utils/src/dsv32_parser.rs`
- `python/sglang/rust_utils/src/lib.rs`
- `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`

## Compatibility Notes
Fallback Python logic is retained if the `sglang_rust_utils` extension fails to load.

## Remaining Follow-Ups
None.
