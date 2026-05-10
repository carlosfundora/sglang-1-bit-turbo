# Benchmark Summary

- Before command: `python3 bench_prefix.py`
- After command: `python3 bench_after.py`
- Before timing (Python `prefix_hold`): ~1.03s per 100,000 iterations
- After timing (Rust PyO3 `prefix_hold`): ~0.69s per 100,000 iterations
- Percent change: ~33% latency reduction.
- Notes: The python fallback was completely replaced by a safe Rust string slicing loop bridging over `PyO3`. While `prefix_hold` was simple, it runs per chunk on the reasoning model streaming path, adding a nice little performance edge and completely migrating the logic away from python.
