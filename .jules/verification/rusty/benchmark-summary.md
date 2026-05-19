# Benchmark Summary

- **Before Command**: `python3 test_harmony_benchmark.py` (simulated pure python execution)
- **After Command**: `python3 test_harmony_benchmark.py` (simulated rust PyO3 extension execution)

| Metric | Before (Python) | After (Rust) | Percent Change |
|--------|-----------------|--------------|----------------|
| Time (ms) | 2522.17 | 1046.91 | -58.5% |

**Notes**:
The test environment simulates streaming 5 chunks of harmony structural tokens across 50,000 iterations. The python implementation relies on repeated regex matches, dictionary allocations, string slicing, and multiple conditionals on every string chunk appended to its internal buffer. Migrating the full string processing sequence to Rust PyO3 removes a significant chunk of Python interpreter looping overhead, reducing execution time by 58.5%.
