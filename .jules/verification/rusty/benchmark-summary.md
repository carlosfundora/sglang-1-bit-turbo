# Benchmark Summary

* Before command: `python3 test_trim_overlap.py`
* After command: `python3 test_trim_overlap_rust.py`
* Before timing: 4902.8 ms
* After timing: 520.1 ms
* Percent change: -89.4%
* Notes on variance or limitations: The rust rewrite is ~10x faster after fixing UTF-8 encoding checking. The PyO3 rust rewrite operates on strings closer to the metal and avoids the constant GC and object allocation overheads of python string slicing inside the loop, while now correctly preventing panics by verifying utf-8 byte character boundaries using `is_char_boundary(i)`.
