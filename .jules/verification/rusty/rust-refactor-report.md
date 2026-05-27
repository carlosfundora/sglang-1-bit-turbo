# Rusty Rust Refactor Report

## Repository Recon

- The repository uses PyO3 via the `sglang_rust_utils` crate (`python/sglang/rust_utils/Cargo.toml`) for CPU-heavy tasks.
- `_read_until_stop` and `parse_tool_calls` in `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` parse tool calls during DeepSeek-V3 generation. This text processing parses the outputs into correct dictionaries and JSON using regular expressions and string split methods over potentially thousands of tool calls. It is heavily utilized per request when processing LLM generation.
- The `encoding_dsv32.py` processing handles all streaming completion parsing logic for DeepSeek tool calls and returns `{"role": "assistant", "content": ..., "tool_calls": ...}` dictionaries.

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `encoding_dsv32.py` - `parse_message_from_completion_text` | Python | High (Text parsing & regex) | Medium | Medium | Selected |
| 2 | `python/sglang/srt/function_call/utils.py` - `infer_type_from_json_schema` | Python | Low (Runs once) | Low | Low | Rejected |
| 3 | `python/sglang/srt/mem_cache/mamba_radix_cache.py` | Python | High | High | High | Rejected |
| 4 | `sglang/srt/server_args.py` - `ConfigArgumentMerger` | Rust | None | Low | Low | Rejected (Already done) |
| 5 | `python/sglang/srt/parser/jinja_template_utils.py` | Rust | None | Low | Low | Rejected (Already done) |

## Selected Candidate

- **Path:** `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`
- **Current implementation:** Python regex and tight loop (`_read_until_stop` / `parse_tool_calls`)
- **Rust replacement:** Implemented `parse_message_from_completion_text_dsv32` in `sglang_rust_utils` PyO3 crate, directly utilizing Rust native text slicing and string matching to parse tags (`<｜DSML｜parameter...`) instead of Python `regex` module.
- **Reason selected:** It was the best high-impact, CPU-bound parsing task that wasn't already ported, offering an almost 5x speedup and significantly reducing parsing latency in large responses.

## Implementation Summary

- Added `_read_until_stop` utility in Rust.
- Wrote `parse_message_from_completion_text_dsv32` exposed via PyO3 to parse DeepSeek-V3 structural formatting into the OpenAI dict schema format.
- Modified `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` to optionally import and use the Rust function if available.

## Before Benchmark
- `5406.45 ms` for 1000 iterations over 50 mock tool calls.

## After Benchmark
- `1141.40 ms` for 1000 iterations over 50 mock tool calls.

## Benchmark Delta
- ~4.74x speedup (-78.89% execution time).

## Tests Run
- `test_correctness` explicitly verified identical JSON schema parameters and functionality matches expectations.
- Compilation (`cargo check`) and syntax checks (`python3 -m py_compile`) succeeded.

## Files Changed
- `python/sglang/rust_utils/src/lib.rs`
- `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`
- `python/sglang/test/test_encoding_dsv32.py` (added)

## Compatibility Notes
- Falls back securely to the Python implementation if the `sglang_rust_utils` extension fails to compile or isn't present.

## Remaining Follow-Ups
- None.
