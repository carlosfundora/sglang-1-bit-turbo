with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "r") as f:
    content = f.read()

# We will just disable the E402 check using # noqa: E402
import_block = """try:
    from sglang.sglang_rust_utils import mamba_match_prefix
    RUST_UTILS_AVAILABLE = True
except ImportError:
    RUST_UTILS_AVAILABLE = False
"""

# Let's add `# ruff: noqa: E402` to the top of the file!
content = "# ruff: noqa: E402\n" + content

lines = content.split('\n')
for i, line in enumerate(lines):
    if line.startswith("import heapq"):
        lines.insert(i, import_block)
        break

content = "\n".join(lines)

split_node_old = """    def _split_node(self, key, node: TreeNode) -> Tuple[TreeNode, int]:
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
        return new_node, match_len"""

split_node_new = """    def _split_node(self, key, node: TreeNode) -> Tuple[TreeNode, int]:
        if RUST_UTILS_AVAILABLE:
            match_len = mamba_match_prefix(node.key, key)
        else:
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
        return new_node, match_len"""

content = content.replace(split_node_old, split_node_new)

match_prefix_old = """    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key

        value = []
        last_node = [self.root_node]
        node = self.root_node

        while len(key) > 0 and key[0] in node.children:
            node = node.children[key[0]]

            prefix_len = self.key_match_fn(node.key, key)
            if prefix_len < len(node.key):
                if prefix_len > 0:
                    last_node[0] = node
                    value.append(node.value[:prefix_len])
                break

            key = key[prefix_len:]
            last_node[0] = node
            value.append(node.value)

        return MatchResult(value, last_node[0])"""

match_prefix_new = """    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
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

        return MatchResult(value, last_node[0])"""

content = content.replace(match_prefix_old, match_prefix_new)

with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "w") as f:
    f.write(content)
