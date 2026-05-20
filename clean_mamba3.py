with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "r") as f:
    content = f.read()

import_block = """
try:
    from sglang.sglang_rust_utils import mamba_match_prefix
    RUST_UTILS_AVAILABLE = True
except ImportError:
    RUST_UTILS_AVAILABLE = False
"""

lines = content.split('\n')
for i, line in enumerate(lines):
    if line.startswith("import heapq"):
        lines.insert(i, import_block.strip() + "\n")
        break

content = "\n".join(lines)
content = "# ruff: noqa: E402\n" + content

# Oh! There are actually 3 places where it's used. Let's fix line 604 and 1220 and 1269.
# The error was happening because I replaced line 604 incorrectly since it wasn't the exact prefix!
# Line 604 is in _insert_helper
# Line 1220 is in _split_node
# Line 1269 is in match_prefix

# Let's just patch all of them properly with inline if expressions to avoid indentation hell!
content = content.replace("match_len = self.key_match_fn(node.key, key)",
                          "match_len = mamba_match_prefix(node.key, key) if RUST_UTILS_AVAILABLE else self.key_match_fn(node.key, key)")
content = content.replace("prefix_len = mamba_match_prefix(list(node.key), list(key)) if RUST_UTILS_AVAILABLE else self.key_match_fn(node.key, key)",
                          "prefix_len = mamba_match_prefix(node.key, key) if RUST_UTILS_AVAILABLE else self.key_match_fn(node.key, key)")

with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "w") as f:
    f.write(content)
