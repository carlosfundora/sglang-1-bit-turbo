with open("test_final_bench.py", "r") as f:
    content = f.read()

content = content.replace("self._match_pre_processor = lambda params: params.key", "self._match_pre_processor = lambda params: params.key\n        self._match_prefix_helper = mamba_radix_cache.MambaRadixCache._match_prefix_helper.__get__(self)")

with open("test_final_bench.py", "w") as f:
    f.write(content)
