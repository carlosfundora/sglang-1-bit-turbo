with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    content = f.read()

import re
# Oh wait, my revert script didn't revert the old mamba_match_prefix function! I appended to the file again!
# Let's cleanly reset lib.rs
