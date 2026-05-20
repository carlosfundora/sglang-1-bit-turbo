# Rust Refactor Plan: MambaRadixCache `_insert_helper` & `_split_node`

## Candidate Selection
`MambaRadixCache` in `python/sglang/srt/mem_cache/mamba_radix_cache.py` (`_insert_helper` and `_split_node`).

## Implementation Steps

1. *Write and run the pre-refactor benchmark script*
   - Use `run_in_bash_session` to create `bench_mamba.py` to benchmark `_insert_helper` and `_split_node` logic.
   - Use `run_in_bash_session` to verify `bench_mamba.py` content using `cat`.
   - Use `run_in_bash_session` to run the script and save the output to `.jules/verification/rusty/before-benchmark.json`.
   - Use `run_in_bash_session` to verify `before-benchmark.json` content using `cat`.
2. *Implement the Rust core*
   - Use `run_in_bash_session` to append `MambaRadixTree` and its logic to `python/sglang/rust_utils/src/lib.rs`.
3. *Verify and compile the Rust extension*
   - Verify the edits by using `run_in_bash_session` to `cat python/sglang/rust_utils/src/lib.rs`.
   - Use `run_in_bash_session` to run `cargo build --release` in `python/sglang/rust_utils/`.
   - Use `run_in_bash_session` to copy `libsglang_rust_utils.so` to `python/sglang/sglang_rust_utils.so`.
4. *Update Python integration*
   - Use `run_in_bash_session` to modify `python/sglang/srt/mem_cache/mamba_radix_cache.py` to use the Rust structures.
   - Use `run_in_bash_session` to verify the modifications with `cat`.
5. *Run tests*
   - Use `run_in_bash_session` to create a `run_test.py` script containing the necessary mocks for testing `test/registered/unit/mem_cache/test_mamba_unittest.py`.
   - Use `run_in_bash_session` to verify `run_test.py` content using `cat`.
   - Use `run_in_bash_session` to execute `python3 run_test.py`.
6. *Run the post-refactor benchmark*
   - Use `run_in_bash_session` to modify `bench_mamba.py` to output to `.jules/verification/rusty/after-benchmark.json`.
   - Use `run_in_bash_session` to verify `bench_mamba.py` modification with `cat`.
   - Use `run_in_bash_session` to run `python3 bench_mamba.py`.
   - Use `run_in_bash_session` to verify `after-benchmark.json` content using `cat`.
   - Use `run_in_bash_session` to write a Python script `calc_speedup.py` to calculate speedup and generate `.jules/verification/rusty/benchmark-summary.md`.
   - Use `run_in_bash_session` to verify `calc_speedup.py` content using `cat`.
   - Use `run_in_bash_session` to execute `python3 calc_speedup.py`.
   - Use `run_in_bash_session` to verify `benchmark-summary.md` content using `cat`.
7. *Document the refactor*
   - Use `run_in_bash_session` to generate `.jules/verification/rusty/rust-refactor-report.md`.
   - Use `run_in_bash_session` to verify `rust-refactor-report.md` content using `cat`.
8. *Run linters and final tests*
    - Use `run_in_bash_session` to run `python3 run_test.py` again to ensure tests pass before submit.
    - Use `run_in_bash_session` to run `ruff check python/sglang/srt/mem_cache/mamba_radix_cache.py`.
9. *Complete pre commit steps*
    - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
10. *Finalize code and Submit*
    - Call the `submit` tool to finalize the code.
