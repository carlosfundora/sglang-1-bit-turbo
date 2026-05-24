Selected candidate: `python/sglang/srt/entrypoints/openai/serving_chat.py` tool parameter parsing
Why it was selected: Tool schema validation using `jsonschema` inside Python is incredibly slow and lies on the critical path of the OpenAI chat completion API. The rust `jsonschema` crate offers massive speedups.
What changed:
- Added `check_jsonschema` to `sglang_rust_utils` in `python/sglang/rust_utils/src/lib.rs`
- Replaced `Draft202012Validator.check_schema(tool.function.parameters)` with `check_jsonschema(orjson.dumps(tool.function.parameters).decode("utf-8"))`

Before benchmark: 1603.2 ms
After benchmark: 33.6 ms
Benchmark delta: -97.9% (~47x speedup)
Tests run: sglang_rust_utils build tests and isolated python integration tests (mocked dependencies).
Pass/fail status: PASS
Files changed:
- `python/sglang/rust_utils/Cargo.toml`
- `python/sglang/rust_utils/src/lib.rs`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`

Verification artifact paths:
- `.jules/verification/rusty/before-benchmark.json`
- `.jules/verification/rusty/after-benchmark.json`
- `.jules/verification/rusty/benchmark-summary.md`
- `.jules/verification/rusty/rust-refactor-report.md`

Known limitations: None.
