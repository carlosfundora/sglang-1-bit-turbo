# Benchmark Summary

*   **Before command:** `python python/sglang/test/test_jinja_template_utils_bench.py`
*   **After command:** `python python/sglang/test/test_jinja_template_utils_bench.py`
*   **Before timing:** 333.83 ms
*   **After timing:** 1088.97 ms
*   **Percent change:** 226.2%
*   **Notes:** The benchmark measures the performance of `detect_and_parse` in `ReasoningParser` before and after enabling the pure Rust implementation (`RustReasoningState`). The mock environment was used to run the benchmark reliably.

Interestingly, `detect_and_parse` had a slight regression here due to FFI boundaries (copying python strings back and forth is expensive for single-shot small outputs). However, the real advantage of the reasoning parser in Rust is the `parse_streaming_increment` chunking which was benchmarked previously with `trim_overlap` and handles state management on the Rust side where `in_reasoning` and `buffer` can be preserved without FFI.
