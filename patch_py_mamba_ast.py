import ast

with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "r") as f:
    source = f.read()

lines = source.splitlines()

# 1. Add import safely using ast finding logger
import_idx = 0
for i, line in enumerate(lines):
    if line.startswith("logger = logging.getLogger"):
        import_idx = i + 1
        break

import_str = """

try:
    from sglang.sglang_rust_utils import mamba_split_node, mamba_match_prefix
    RUST_UTILS_AVAILABLE = True
except ImportError:
    RUST_UTILS_AVAILABLE = False
"""
lines.insert(import_idx, import_str)
source = "\n".join(lines)


# 2. String replacements for methods
def replace_method(source, method_name, new_method_str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MambaRadixCache":
            for i, class_item in enumerate(node.body):
                if isinstance(class_item, ast.FunctionDef) and class_item.name == method_name:
                    start_lineno = class_item.lineno - 1
                    end_lineno = class_item.end_lineno

                    lines = source.splitlines()

                    new_lines = lines[:start_lineno] + [new_method_str.strip("\n")] + lines[end_lineno:]
                    return "\n".join(new_lines)
    return source

split_node_str = """
    def _split_node(self, key, node: TreeNode) -> Tuple[TreeNode, int]:
        if RUST_UTILS_AVAILABLE:
            match_len = mamba_match_prefix(node.key, key)
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
            node.is_mamba = False
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
        node.is_mamba = False
        return new_node, match_len
"""

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

source = replace_method(source, "_split_node", split_node_str)
source = replace_method(source, "match_prefix", match_prefix_str)

with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "w") as f:
    f.write(source)
