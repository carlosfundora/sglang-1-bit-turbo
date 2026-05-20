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

# Instead of blindly replacing prefix_len which breaks indentation, I will do exact string matches!
content = content.replace(
"""    def _split_node(self, key, node: TreeNode) -> Tuple[TreeNode, int]:
        match_len = self.key_match_fn(node.key, key)""",
"""    def _split_node(self, key, node: TreeNode) -> Tuple[TreeNode, int]:
        if RUST_UTILS_AVAILABLE:
            match_len = mamba_match_prefix(node.key, key)
        else:
            match_len = self.key_match_fn(node.key, key)"""
)

content = content.replace(
"""        while len(key) > 0 and key[0] in node.children:
            node = node.children[key[0]]

            prefix_len = self.key_match_fn(node.key, key)""",
"""        while len(key) > 0 and key[0] in node.children:
            node = node.children[key[0]]

            if RUST_UTILS_AVAILABLE:
                prefix_len = mamba_match_prefix(node.key, key)
            else:
                prefix_len = self.key_match_fn(node.key, key)"""
)

with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "w") as f:
    f.write(content)
