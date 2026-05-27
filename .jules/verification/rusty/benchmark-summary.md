# DSV32 Rust Refactor Benchmark

- **Before command:** `python python/sglang/test/test_encoding_dsv32.py before`
- **After command:** `python python/sglang/test/test_encoding_dsv32.py after`
- **Before timing:** 5406.45 ms (1000 iterations, 50 tool calls)
- **After timing:** 1141.40 ms (1000 iterations, 50 tool calls)
- **Percent change:** -78.89% (or ~4.7x speedup)
- **Notes on variance or limitations:** The string parsing for parameter keys and tool names was converted from regex and python loops to direct string slices natively in Rust.
