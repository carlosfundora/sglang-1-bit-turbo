# Rust Refactor Candidates

1. `python/sglang/srt/parser/jinja_template_utils.py`: `detect_jinja_template_content_format` & `process_content_for_template_format`
   - **Current runtime:** Python
   - **What it does:** Parses chat templates to figure out whether the output should be "string" or "openai" structured messages, extracting variables and AST elements using `jinja2`.
   - **Why Rust:** Text/AST manipulation and checking for nested elements is well-suited for Rust.
   - **Expected benefit:** Speed up prompt pre-processing, which happens for every request.
   - **Estimated complexity:** High (requires integrating Jinja AST parsing or a regex-based approximation into Rust, plus PyO3 bindings for dict/ImageData).
   - **Risk level:** Medium.

2. `python/sglang/srt/layers/utils/hash.py`: `murmur_hash32`
   - **Current runtime:** Python/Triton
   - **What it does:** Murmur3 hash implementation used inside kernel launches.
   - **Why Rust:** The Triton jit compiles to GPU, so converting to Rust would only be for CPU-side hashing if needed, but it seems to be primarily for GPU kernels. Not a good candidate.

3. `python/sglang/srt/parser/reasoning_parser.py`: Several Reasoning Parsers (`DeepSeekR1Detector`, `Qwen3Detector`, etc)
   - **Current runtime:** Python
   - **What it does:** Parses string tokens as they stream in to split out `<think>` vs normal text sections.
   - **Why Rust:** Tight string processing loop run on every chunk of generated output.
   - **Expected benefit:** Lower overhead for parsing logic on the CPU during the streaming phase.
   - **Estimated complexity:** Low-Medium.
   - **Risk level:** Low.
   - **Test strategy:** Write unit tests to check state transitions of the parser.

4. `sglang/srt/server_args.py`: Command Line argument parsing.
   - Already has `ConfigArgumentMerger` refactored to Rust!

5. `python/sglang/rust_utils/src/lib.rs` / `trim_overlap`:
   - It's already in Rust!

6. `python/sglang/srt/mem_cache/mamba_radix_cache.py`: `_insert_helper`, `_split_node`
   - **Current runtime:** Python
   - **What it does:** Manages a radix tree for prefix caching.
   - **Why Rust:** Graph traversal, frequent allocations, and tree mutations in Python are slow.
   - **Expected benefit:** Faster cache insertions and cache hit lookups.
   - **Estimated complexity:** High (they already ported the radix tree to C++ in `cpp_radix_tree`, so it might be redundant or conflict with existing C++ code).
   - **Risk level:** High.
