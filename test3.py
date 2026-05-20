with open("python/sglang/srt/mem_cache/mamba_radix_cache.py", "r") as f:
    content = f.read()

import_block = """try:
    from sglang.sglang_rust_utils import mamba_match_prefix
    RUST_UTILS_AVAILABLE = True
except ImportError:
    RUST_UTILS_AVAILABLE = False
"""

# Place it exactly after `import heapq`
# Use regex to properly insert it
lines = content.split('\n')
for i, line in enumerate(lines):
    if line.startswith("import torch"):
        lines.insert(i-1, import_block)
        break

# Since ruff keeps complaining about "Module level import not at top of file"
# The solution is simple: Add `# noqa: E402` to ALL imports after our custom try/except
# OR we put the try/except AT THE VERY END OF THE IMPORTS, right before `logger = logging.getLogger(__name__)`
# And then add `# noqa: E402` just to `import logging`

content = "\n".join(lines)

# Just write a clean python script using standard string replacement that will work and be PEP8 compliant
