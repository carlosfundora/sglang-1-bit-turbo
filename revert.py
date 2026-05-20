with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "r") as f:
    content = f.read()

import re

# We remove ALL `try...except ImportError` blocks with `RUST_UTILS_AVAILABLE`
# We completely revert the file back to original python state, and we will rewrite ONLY exactly what is needed correctly.
content = re.sub(r'try:\s*from sglang\.sglang_rust_utils import mamba_match_prefix\s*RUST_UTILS_AVAILABLE = True\s*except ImportError:\s*RUST_UTILS_AVAILABLE = False', '', content)
content = content.replace("# ruff: noqa: E402\n", "")

# Also need to restore the methods to original format. Let's just grab from git from master if we can, but since it's staged we can use `git checkout origin/main -- python/sglang/srt/mem_cache/mamba_radix_cache.py`
