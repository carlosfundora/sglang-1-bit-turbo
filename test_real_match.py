import sys
import types
import os
import time
import importlib.util

sys.path.append(os.path.join(os.getcwd(), 'python'))

spec = importlib.util.spec_from_file_location("sglang_rust_utils", "python/sglang/sglang_rust_utils.so")
sglang_rust_utils = importlib.util.module_from_spec(spec)
sys.modules["sglang.sglang_rust_utils"] = sglang_rust_utils
spec.loader.exec_module(sglang_rust_utils)

start = time.perf_counter()
res = 0
key1 = tuple(range(100))
key2 = tuple(list(range(100)) + [99])

for _ in range(100000):
    res += sglang_rust_utils.mamba_match_prefix(key1, key2)
end = time.perf_counter()

print(f"Rust mamba_match_prefix: {res} matches in {(end-start)*1000:.2f} ms")

def py_match(a, b):
    return sum(1 for x, y in zip(a, b) if x == y)

start = time.perf_counter()
res2 = 0
for _ in range(100000):
    res2 += py_match(key1, key2)
end = time.perf_counter()

print(f"Python py_match: {res2} matches in {(end-start)*1000:.2f} ms")
