# Benchmark Summary: _find_common_prefix

## Before Refactor
- **Command:** `python python/sglang/test/test_find_common_prefix_bench.py before`
- **Timing:** 11327.10 ms (for 5000 iterations on ~9000 character strings)

## After Refactor
- **Command:** `python python/sglang/test/test_find_common_prefix_bench.py after`
- **Timing:** 67.85 ms (for 5000 iterations on same strings)

## Delta
- **Percent Change:** ~99.4% improvement
- **Notes:** The previous python implementation used a python string concatenation in a loop (`prefix += s1[i]`) making the time complexity quadratic `O(N^2)` with respects to the common prefix length. The Rust implementation uses a fast byte comparison loop `s1.bytes().zip(s2.bytes())` and performs character boundary safety checks, resolving the operation in a fraction of a millisecond.
