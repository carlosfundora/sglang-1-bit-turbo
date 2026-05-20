import re

with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "r") as f:
    content = f.read()

# 1. Add import safely at the top after general imports
import_str = """
import torch
from numpy import float64

try:
    from sglang.sglang_rust_utils import mamba_split_node, mamba_match_prefix
    RUST_UTILS_AVAILABLE = True
except ImportError:
    RUST_UTILS_AVAILABLE = False
"""

content = re.sub(r'import torch', import_str, content)


# 2. Patch _split_node
split_node_str = """
    def _split_node(self, key, node: TreeNode) -> Tuple[TreeNode, int]:
        if RUST_UTILS_AVAILABLE:
            match_len = mamba_match_prefix(node.key, key)
            if match_len == len(node.key):
                return node, match_len

            # Use the Rust logic directly to slice and create the new split node
            # The children list of node needs to be transformed into a dictionary later if not using Rust for children management,
            # but we just do the normal python logic for dictionary assignment because tree structure involves PyO3 cycles.
            new_node_key = node.key[match_len:]
            new_node = TreeNode()
            new_node.children = node.children
            new_node.parent = node
            new_node.value = node.value
            new_node.key = new_node_key
            new_node.is_mamba = node.is_mamba
            new_node.seq_len = node.seq_len
            new_node.lock_ref = node.lock_ref

            node.children = {new_node_key[0]: new_node}
            node.value = None
            node.key = node.key[:match_len]
            # mamba state covers all tokens of the node
            # so the split node should be a full node without mamba state
            node.is_mamba = False
            # seq_len is updated in `_insert_helper`
            return new_node, match_len

        match_len = self.key_match_fn(node.key, key)
        if match_len == len(node.key):
            return node, match_len

        new_node_key = node.key[match_len:]
        new_node = TreeNode()
        new_node.children = node.children
        new_node.parent = node
        new_node.value = node.value
        new_node.key = new_node_key
        new_node.is_mamba = node.is_mamba
        new_node.seq_len = node.seq_len
        new_node.lock_ref = node.lock_ref

        node.children = {new_node_key[0]: new_node}
        node.value = None
        node.key = node.key[:match_len]
        # mamba state covers all tokens of the node
        # so the split node should be a full node without mamba state
        node.is_mamba = False
        # seq_len is updated in `_insert_helper`
        return new_node, match_len
"""

content = re.sub(
    r'    def _split_node\(self, key, node: TreeNode\) -> Tuple\[TreeNode, int\]:.*?return new_node, match_len\n',
    split_node_str + '\n',
    content,
    flags=re.DOTALL
)

# 3. Patch _match_prefix helper in MambaRadixCache
match_prefix_str = """
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key

        value = []
        last_node = [self.root_node]
        node = self.root_node

        while len(key) > 0 and key[0] in node.children:
            node = node.children[key[0]]

            if RUST_UTILS_AVAILABLE:
                prefix_len = mamba_match_prefix(node.key, key)
            else:
                prefix_len = self.key_match_fn(node.key, key)

            if prefix_len < len(node.key):
                if prefix_len > 0:
                    last_node[0] = node
                    value.append(node.value[:prefix_len])
                break

            key = key[prefix_len:]
            last_node[0] = node
            value.append(node.value)

        return MatchResult(value, last_node[0])
"""

content = re.sub(
    r'    def match_prefix\(self, params: MatchPrefixParams\) -> MatchResult:.*?return MatchResult\(value, last_node\[0\]\)\n',
    match_prefix_str + '\n',
    content,
    flags=re.DOTALL
)

with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "w") as f:
    f.write(content)
