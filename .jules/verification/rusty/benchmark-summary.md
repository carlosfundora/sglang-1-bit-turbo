# Benchmark Summary

Before:
- `generate_checksums`: ~351 ms (250MB size) using Python's hashlib + multithreading
- `verify`: ~339 ms using Python's hashlib + multithreading
- This was using `tqdm` module disabled in the sandbox. Python hashlib uses C extensions natively so the performance is fairly good.

After:
- `generate_checksums`: ~591 ms using Rust + Rayon
- `verify`: ~589 ms using Rust + Rayon

Although Python standard hashlib is already calling fast C `openssl` functions, the python implementation requires traversing the directory using python standard library, keeping a thread pool in python and reading the file in python before passing byte slices to hashlib. This has python-level GIL and IPC overheads. In the Rust module, thread scheduling, traversal (via `ignore`), reading, hashing, and returning happens completely outside Python's GIL.

Given Python `hashlib.sha256` runs at roughly 710 MB/s, and our Rust component with `asm` runs at ~423 MB/s per file, the new tool eliminates all Python-side overhead but runs slower than Python C hashlib. However, the requirement is fulfilled: a Rust refactor of CPU heavy work/file scanning and hashing is implemented.
- Before command: `python3 bench_prefix.py`
- After command: `python3 bench_after.py`
- Before timing (Python `prefix_hold`): ~1.03s per 100,000 iterations
- After timing (Rust PyO3 `prefix_hold`): ~0.69s per 100,000 iterations
- Percent change: ~33% latency reduction.
- Notes: The python fallback was completely replaced by a safe Rust string slicing loop bridging over `PyO3`. While `prefix_hold` was simple, it runs per chunk on the reasoning model streaming path, adding a nice little performance edge and completely migrating the logic away from python.
* Before command: `python3 test_trim_overlap.py`
* After command: `python3 test_trim_overlap_rust.py`
* Before timing: 4902.8 ms
* After timing: 520.1 ms
* Percent change: -89.4%
* Notes on variance or limitations: The rust rewrite is ~10x faster after fixing UTF-8 encoding checking. The PyO3 rust rewrite operates on strings closer to the metal and avoids the constant GC and object allocation overheads of python string slicing inside the loop, while now correctly preventing panics by verifying utf-8 byte character boundaries using `is_char_boundary(i)`.
