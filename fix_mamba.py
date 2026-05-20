with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "r") as f:
    lines = f.readlines()

out = []
skip = False
for line in lines:
    if "from sglang.srt.mem_cache.base_prefix_cache import (" in line: skip = True; continue
    if "from sglang.srt.mem_cache.allocator import (" in line: skip = True; continue
    if skip and ")" in line: skip = False; continue
    if skip: continue

    if line.startswith("from sglang.srt."):
        continue
    out.append(line)

with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "w") as f:
    f.writelines(out)
