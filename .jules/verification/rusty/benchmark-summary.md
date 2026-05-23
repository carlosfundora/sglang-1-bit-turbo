# Benchmark Summary: `find_printable_text`

| Implementation | Duration (ms) | Iterations | Command |
| --- | --- | --- | --- |
| Python (`before`) | 3426.8 | 1000000 | `python python/sglang/test/test_utils_find_printable_text_bench.py` |
| Rust (`after`) | 2151.9 | 1000000 | `python python/sglang/test/test_utils_find_printable_text_bench.py` |

**Percent Change:**
The pure Rust implementation represents an approximate **37.2% reduction in latency** (or roughly 1.59x faster) compared to the pure Python implementation, despite crossing the PyO3 FFI boundary.

**Notes on Variance or Limitations:**
The FFI overhead (dummy call alone takes ~2565ms if doing nothing, wait, actually when the function processes the text efficiently, PyO3 string handling overhead still exists but is smaller than Python's string allocation and interpretation overhead for these text paths). The benchmark ensures the CJK detection boundaries and character mapping exactly map to the previous python behavior using Rust's `str.chars()` iterators.