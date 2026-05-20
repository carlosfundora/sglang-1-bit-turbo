with open("test_final_bench.py", "r") as f:
    content = f.read()

content = content.replace("self._match_prefix_helper = mamba_radix_cache.MambaRadixCache._match_prefix_helper.__get__(self)", "self._match_prefix_helper = mamba_radix_cache.MambaRadixCache._match_prefix_helper.__get__(self)\n        self.get_child_key_fn = lambda x: x[0]")

with open("test_final_bench.py", "w") as f:
    f.write(content)
