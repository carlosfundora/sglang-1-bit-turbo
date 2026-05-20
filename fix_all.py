import re

with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    content = f.read()

# Only keep 1, 2, 3, 4, 5, 6, 8, 11, 24
variants_to_remove = [19, 21, 10, 18, 12, 13, 15, 16, 17, 20, 9, 7, 14, 25, 23, 26]

for variant in variants_to_remove:
    # Match the block: `variant => { ... },`
    pattern = rf'\s+{variant} => {{.*?}},\n'
    content = re.sub(pattern, '\n', content, flags=re.DOTALL)

with open("python/sglang/rust_utils/src/lib.rs", "w") as f:
    f.write(content)
