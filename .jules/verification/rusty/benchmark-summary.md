# Benchmark Summary

- **Candidate:** `python/sglang/srt/parser/jinja_template_utils.py`
- **Functions:** `process_content_for_template_format` (Note: `detect_jinja_template_content_format` is left in Python due to AST parsing constraints)
- **Before Command:** `python3 test_jinja2_both.py` (Using Python `dict` iteration and lists processing)
- **After Command:** `python3 test_jinja2_both_rust.py` (Using Rust PyO3 Extension)
- **Before Timing:** 56.035s (per 100,000 iterations for both detect and process)
- **After Timing:** 1.405s (per 100,000 iterations for both detect and process)
- **Percent Change:** -97.5% runtime (~39x speedup)
- **Notes on Variance or Limitations:** The benchmark was run inside a mocked environment.
