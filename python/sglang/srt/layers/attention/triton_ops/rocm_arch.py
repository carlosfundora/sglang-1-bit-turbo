
import torch
from sglang.srt.utils import is_hip

_CACHED_GCN_ARCH = None

def get_gcn_arch():
    global _CACHED_GCN_ARCH
    if _CACHED_GCN_ARCH is None:
        if is_hip():
            try:
                _CACHED_GCN_ARCH = torch.cuda.get_device_properties(0).gcnArchName
            except Exception:
                _CACHED_GCN_ARCH = ""
        else:
            _CACHED_GCN_ARCH = ""
    return _CACHED_GCN_ARCH

def is_gfx1030():
    arch = get_gcn_arch()
    return arch is not None and "gfx103" in arch
