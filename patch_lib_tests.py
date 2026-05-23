import re

with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    content = f.read()

# Fix the trims_overlap bug that was already there.
# It failed because of slicing on invalid utf8 boundaries.
old_str = """
    for i in 1..=max_possible {
        if existing_text.ends_with(&new_chunk[..i]) {
            max_overlap = i;
        }
    }
"""

new_str = """
    for i in 1..=max_possible {
        if new_chunk.is_char_boundary(i) && existing_text.ends_with(&new_chunk[..i]) {
            max_overlap = i;
        }
    }
"""

content = content.replace(old_str, new_str)

with open("python/sglang/rust_utils/src/lib.rs", "w") as f:
    f.write(content)
