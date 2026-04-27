import re

with open("python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py", "r") as f:
    content = f.read()

pattern = """from sglang.srt.utils import is_hip
from sglang.srt.hardware_backend.rocm.arch_detection import is_rdna2

_is_hip = is_hip()
_is_rdna2 = is_hip() and is_rdna2()

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _decode_softmax_reducev_fwd,
)"""

replacement = """from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _decode_softmax_reducev_fwd,
)
from sglang.srt.hardware_backend.rocm.arch_detection import is_rdna2
from sglang.srt.utils import is_hip

_is_hip = is_hip()
_is_rdna2 = is_hip() and is_rdna2()"""

content = content.replace(pattern, replacement)

with open("python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py", "w") as f:
    f.write(content)
