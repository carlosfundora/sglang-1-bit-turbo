# SGLang public APIs

import os as _os
import pathlib as _pathlib
import sys as _sys

# ROCm gfx1030 workaround (AMD RX 6700 XT with HSA_OVERRIDE_GFX_VERSION=10.3.0)
# MUST run BEFORE any torch import to patch torch.zeros/ones/fill_ before torch caches pointers
try:
    import universal_kv.hip_zero_patch as _hip_zero_patch
    _hip_zero_patch.apply()
    if _os.environ.get("SGLANG_STRICT_PROVENANCE") == "1":
        _repo_root = _pathlib.Path(__file__).resolve().parents[2]
        _patch_path = _pathlib.Path(_hip_zero_patch.__file__).resolve()
        if not str(_patch_path).startswith(str(_repo_root)):
            raise RuntimeError(
                "hip_zero_patch provenance mismatch: "
                f"{_patch_path} is outside {_repo_root}. "
                "Use a single source tree for sglang + universal_kv."
            )
except ImportError:
    if _os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0":
        raise RuntimeError(
            "Failed to load universal_kv.hip_zero_patch while HSA_OVERRIDE_GFX_VERSION=10.3.0. "
            "This configuration requires the patch to avoid ROCm fill-kernel segfaults."
        )
    pass  # patch not required when gfx1030 override is not active

# Install stubs early for platforms where certain dependencies are unavailable
# (e.g. macOS/MPS has no triton, and torch.mps lacks Stream / set_device /
# get_device_properties).  This must run before any downstream imports.
if _sys.platform == "darwin":
    try:
        import torch as _torch

        if _torch.backends.mps.is_available():
            from sglang._triton_stub import install as _install_triton_stub

            _install_triton_stub()
            del _install_triton_stub

            from sglang._mps_stub import install as _install_mps_stub

            _install_mps_stub()
            del _install_mps_stub
        del _torch
    except ImportError:
        pass

del _os
del _pathlib
del _sys

# Frontend Language APIs
from sglang.global_config import global_config
from sglang.lang.api import (
    Engine,
    Runtime,
    assistant,
    assistant_begin,
    assistant_end,
    flush_cache,
    function,
    gen,
    gen_int,
    gen_string,
    get_server_info,
    image,
    select,
    separate_reasoning,
    set_default_backend,
    system,
    system_begin,
    system_end,
    user,
    user_begin,
    user_end,
    video,
)
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.choices import (
    greedy_token_selection,
    token_length_normalized,
    unconditional_likelihood_normalized,
)

# Lazy import some libraries
from sglang.utils import LazyImport
from sglang.version import __version__

Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
LiteLLM = LazyImport("sglang.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("sglang.lang.backend.vertexai", "VertexAI")

# Runtime Engine APIs
ServerArgs = LazyImport("sglang.srt.server_args", "ServerArgs")
Engine = LazyImport("sglang.srt.entrypoints.engine", "Engine")

__all__ = [
    "Engine",
    "Runtime",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_info",
    "image",
    "select",
    "separate_reasoning",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "user",
    "user_begin",
    "user_end",
    "video",
    "RuntimeEndpoint",
    "greedy_token_selection",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
    "ServerArgs",
    "Anthropic",
    "LiteLLM",
    "OpenAI",
    "VertexAI",
    "global_config",
    "__version__",
]
