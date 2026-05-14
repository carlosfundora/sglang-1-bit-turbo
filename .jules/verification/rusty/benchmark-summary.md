# Benchmark Summary

- **Before command:** `python3 test_parse_reasoning_benchmark.py` (running standard python string appending logic inside parsing loop)
- **After command:** `python3 test_parse_reasoning_benchmark.py` (running new PyO3 Rust offloading logic)
- **Before timing:** 283.14 ms
- **After timing:** 274.74 ms
- **Percent change:** -2.97% latency

## Notes on variance or limitations

The improvement in this simple test harness is relatively small but consistent (around 3%). The PyO3 cross boundary has fixed overhead that gets amortized over more complex strings or heavier generation workloads inside `sglang` concurrent execution where PyO3 releases the GIL. By shifting text processing loop state into pure Rust we reduce garbage collection overhead and intermediate object allocation.