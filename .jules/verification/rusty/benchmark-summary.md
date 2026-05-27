# Benchmark Summary

- **Before Command**: `python test_dsv32_minimal.py`
- **After Command**: `python test_rust_parser.py`
- **Before Timing**: 680.70 ms (10,000 iterations)
- **After Timing**: 139.82 ms (10,000 iterations)
- **Percent Change**: 79.5% improvement (~4.8x speedup)
- **Notes**: The pure Rust parser is vastly faster by avoiding Python regular expressions (`re.findall`) and heavy string slicing loops in the inner parsing of `parse_message_from_completion_text`. The Rust implementation also directly creates the resulting `PyDict` bypassing intermediate Python object allocations.
