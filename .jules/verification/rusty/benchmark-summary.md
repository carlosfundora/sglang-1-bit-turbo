# MambaRadixCache Benchmark Summary

| Metric | Before Refactor | After Refactor | Change |
|---|---|---|---|
| Command | `python3 bench_mamba.py` | `python3 bench_mamba.py` | - |
| Workload | 100,000 node matches | 100,000 node matches | - |
| Duration | 2912.18 ms | 744.17 ms | -74.45% |
| Throughput | 34338.54 req/s | 134377.90 req/s | +291.33% |

**Notes:**
The bottleneck was previously the `zip(a, b)` generator logic inside `self._key_match_page_size1`. Moving this logic to a stateless Rust helper (`mamba_match_prefix`) via `PyO3` allows it to loop over integer slices at native C speed without instantiating Python tuples or invoking the CPython interpreter loop.
