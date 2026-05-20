with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "r") as f:
    content = f.read()

import_block = """
try:
    from sglang.sglang_rust_utils import mamba_match_prefix
    RUST_UTILS_AVAILABLE = True
except ImportError:
    RUST_UTILS_AVAILABLE = False
"""

# Insert perfectly at top
lines = content.split('\n')
for i, line in enumerate(lines):
    if line.startswith("import heapq"):
        lines.insert(i, import_block.strip() + "\n")
        break

content = "\n".join(lines)
content = "# ruff: noqa: E402\n" + content

# Replace the specific lines inside _split_node
split_old = "match_len = self.key_match_fn(node.key, key)"
split_new = """if RUST_UTILS_AVAILABLE:
            match_len = mamba_match_prefix(node.key, key)
        else:
            match_len = self.key_match_fn(node.key, key)"""

# In _split_node
content = content.replace(
    "    def _split_node(self, key, node: TreeNode) -> Tuple[TreeNode, int]:\n        match_len = self.key_match_fn(node.key, key)",
    "    def _split_node(self, key, node: TreeNode) -> Tuple[TreeNode, int]:\n        " + split_new
)

# In match_prefix
content = content.replace(
    "            prefix_len = self.key_match_fn(node.key, key)",
    "            " + split_new.replace("        ", "            ")
)

with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "w") as f:
    f.write(content)
