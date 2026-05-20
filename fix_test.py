import re

with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    content = f.read()

# Wait, `LLAMA4` is index 9. But in my rust implementation I had `9 => LLAMA4` AND I NEVER DELETED IT!
# Let me check `python/sglang/rust_utils/src/lib.rs`
