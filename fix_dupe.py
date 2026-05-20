import re

with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    content = f.read()

# We double-pasted conversation_get_prompt, let's remove everything after the first `#[pymodule]` because that's what we appended previously by mistake.
idx = content.find("#[pymodule]")
if idx != -1:
    end_idx = content.find("Ok(())", idx)
    end_idx = content.find("}", end_idx) + 1
    content = content[:end_idx] + "\n"

with open("python/sglang/rust_utils/src/lib.rs", "w") as f:
    f.write(content)
