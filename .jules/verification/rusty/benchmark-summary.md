# Benchmark Summary

- **Before command:** `python python/sglang/test/test_longbench_v2_bench.py`
- **After command:** `python python/sglang/test/test_longbench_v2_bench.py`
- **Before timing:** 211.11 ms
- **After timing:** 112.44 ms
- **Percent change:** -46.74% (faster)
- **Notes on variance or limitations:** The string processing is small, so FFI overhead is present, but using pre-compiled `OnceLock` Regex in Rust allows the execution to be roughly twice as fast as the Python execution on identical inputs for 100,000 iterations.
