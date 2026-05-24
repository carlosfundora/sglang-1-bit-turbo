# Benchmark Summary

- Before Command: `python3 .jules/verification/rusty/bench_jsonschema.py`
- After Command: `python3 .jules/verification/rusty/bench_jsonschema_rust.py`
- Before Timing: 1603.2 ms
- After Timing: 33.6 ms
- Percent Change: -97.9% (~47x speedup)
- Notes: Validating JSON schema in Python using `jsonschema` is incredibly slow, compiling this check to PyO3 using the Rust `jsonschema` and `serde_json` crate provides significant speedups during hotpath tool verification.
