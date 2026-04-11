# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.batch_invariant_ops import (
    is_batch_invariant_mode_enabled,
    rms_norm_batch_invariant,
)
from sglang.srt.environ import envs
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_flashinfer_layernorm_available = False

# RDNA detection — AITER CK-based rmsnorm JIT fails on RDNA architectures
_is_rdna_for_layernorm = False
if _is_hip and _use_aiter:
    try:
        _props = torch.cuda.get_device_properties(0)
        _arch = getattr(_props, "gcnArchName", "").split(":")[0]
        _is_rdna_for_layernorm = _arch.startswith("gfx10") or _arch.startswith("gfx11") or _arch.startswith("gfx12")
        if _is_rdna_for_layernorm:
            logging.getLogger(__name__).info(
                f"RDNA GPU ({_arch}): AITER CK-based rmsnorm bypassed, using forward_hip chain"
            )
    except Exception:
        pass

# RDNA2 Wave32 HIP kernel — lazy-init (avoids JIT compile at import time)
_rdna2_rmsnorm_checked = False
_rdna2_rmsnorm_ok = False
_rdna2_fused_add_rms_norm_cached = None
_rdna2_rms_norm_cached = None


def _check_rdna2_rmsnorm():
    """Lazy one-time check for RDNA2 RMSNorm kernel availability."""
    global _rdna2_rmsnorm_checked, _rdna2_rmsnorm_ok
    global _rdna2_fused_add_rms_norm_cached, _rdna2_rms_norm_cached
    if _rdna2_rmsnorm_checked:
        return _rdna2_rmsnorm_ok
    _rdna2_rmsnorm_checked = True
    if not _is_hip:
        return False
    try:
        from sglang.srt.layers.kernels.rdna2.dispatch import rdna2_ops

        _rdna2_rmsnorm_ok = rdna2_ops.probe() and os.environ.get("SGLANG_RDNA2_RMSNORM", "1") != "0"
        if _rdna2_rmsnorm_ok:
            from sglang.srt.layers.kernels.rdna2.rmsnorm import (
                fused_add_rms_norm as rdna2_fused_add_rms_norm,
                rms_norm as rdna2_rms_norm,
            )
            _rdna2_fused_add_rms_norm_cached = rdna2_fused_add_rms_norm
            _rdna2_rms_norm_cached = rdna2_rms_norm
            logger.info("RDNA2 Wave32 RMSNorm: enabled for forward_hip dispatch")
    except Exception:
        _rdna2_rmsnorm_ok = False
    return _rdna2_rmsnorm_ok
    _rdna2_rmsnorm_checked = True
    if not _is_hip:
        return False
    try:
        from sglang.srt.layers.kernels.rdna2.dispatch import rdna2_ops

        _rdna2_rmsnorm_ok = rdna2_ops.probe() and os.environ.get("SGLANG_RDNA2_RMSNORM", "1") != "0"
        if _rdna2_rmsnorm_ok:
            logger.info("RDNA2 Wave32 RMSNorm: enabled for forward_hip dispatch")
    except Exception:
        _rdna2_rmsnorm_ok = False
    return _rdna2_rmsnorm_ok

logger = logging.getLogger(__name__)

if _is_cuda or _is_xpu:
    if _is_flashinfer_available:
        try:
            from flashinfer.norm import layernorm

            _flashinfer_layernorm_available = True
        except (ImportError, AttributeError):
            _flashinfer_layernorm_available = False
    else:
        _flashinfer_layernorm_available = False

    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )
_has_vllm_rms_norm = False
_rms_norm_is_inplace = False  # True = vllm 4-arg API, False = aiter/triton 3-arg API
if _use_aiter and not _is_rdna_for_layernorm:
    # RDNA GPUs: skip AITER CK-based rmsnorm import — JIT compilation fails on gfx10xx.
    # RDNA path uses forward_hip (RDNA2 HIP kernels) or Triton fallback instead.
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm

    _has_vllm_rms_norm = True
    _rms_norm_is_inplace = False
elif _is_hip:
    try:
        from vllm._custom_ops import fused_add_rms_norm, rms_norm

        _has_vllm_rms_norm = True
        _rms_norm_is_inplace = True  # vllm: rms_norm(out, x, w, eps) -> None
    except ImportError:
        # Fallback: Triton-based RMSNorm for HIP without aiter/vllm
        # This enables RDNA2 (gfx1030) GPUs to run without MI-series dependencies
        try:
            from sglang.srt.layers.elementwise import fused_rmsnorm

            def rms_norm(x, weight, epsilon):
                return fused_rmsnorm(x, weight, epsilon)

            def fused_add_rms_norm(x, residual, weight, epsilon):
                x = x + residual
                return rms_norm(x, weight, epsilon), x

            _has_vllm_rms_norm = True
            _rms_norm_is_inplace = False  # triton: rms_norm(x, w, eps) -> out
            logger.info(
                "Using Triton RMSNorm fallback for HIP (no aiter/vllm available)"
            )
        except ImportError:
            _has_vllm_rms_norm = False

if _is_npu:
    import torch_npu


def _forward_with_allreduce_fusion(
    norm_module,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    post_residual_addition: Optional[torch.Tensor],
    weight: torch.Tensor,
    use_attn_tp_group: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Shared allreduce-fused RMSNorm logic usable by any norm."""
    if residual is not None:
        from sglang.srt.distributed import (
            get_attn_tensor_model_parallel_world_size,
            get_moe_expert_parallel_world_size,
            get_moe_tensor_parallel_world_size,
            tensor_model_parallel_all_reduce,
            tensor_model_parallel_fused_allreduce_rmsnorm,
        )
        from sglang.srt.layers.flashinfer_comm_fusion import (
            flashinfer_allreduce_residual_rmsnorm,
        )

        if use_attn_tp_group:
            world_size = get_attn_tensor_model_parallel_world_size()
        else:
            if get_moe_expert_parallel_world_size() > 1:
                world_size = get_moe_expert_parallel_world_size()
            else:
                world_size = get_moe_tensor_parallel_world_size()

        if world_size > 1:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition

            # Prefer AITER fused AR+RMSNorm when enabled on AMD (CDNA only).
            if _use_aiter and not _is_rdna_for_layernorm:
                fused_result = tensor_model_parallel_fused_allreduce_rmsnorm(
                    x, residual, weight, norm_module.variance_epsilon
                )
                if fused_result is not None:
                    return fused_result
            else:
                fused_result = flashinfer_allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=weight,
                    eps=norm_module.variance_epsilon,
                    use_attn_tp_group=use_attn_tp_group,
                )
                if fused_result[0] is not None:
                    return fused_result

            # For AITER route, preserve correctness when fused path is unavailable.
            if _use_aiter and not _is_rdna_for_layernorm and get_global_server_args().enable_aiter_allreduce_fusion:
                x = tensor_model_parallel_all_reduce(x)
                return norm_module.forward(x, residual, None)

    return norm_module.forward(x, residual, post_residual_addition)


class RMSNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        cast_x_before_out_mul: bool = False,
        fp32_residual: bool = False,
        has_weight: bool = True,
        weight_dtype: Optional = None,
        override_orig_dtype: Optional = None,
    ) -> None:
        super().__init__()
        self.has_weight = has_weight
        self.cast_x_before_out_mul = cast_x_before_out_mul
        self.fp32_residual = fp32_residual
        self.override_orig_dtype = override_orig_dtype
        if self.has_weight:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
        else:
            self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        if _use_aiter:
            if _is_rdna_for_layernorm:
                # RDNA: AITER CK rmsnorm JIT fails; use forward_hip (RDNA2 kernel)
                self._forward_method = self.forward_hip
            else:
                self._forward_method = self.forward_aiter

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.numel() == 0:
            return x
        if self.variance_size_override is not None:
            return self.forward_native(x, residual, post_residual_addition)
        if is_batch_invariant_mode_enabled():
            if (
                residual is not None
                or get_global_server_args().rl_on_policy_target == "fsdp"
            ):
                return self.forward_native(x, residual, post_residual_addition)
            return rms_norm_batch_invariant(
                x,
                self.weight.data,
                self.variance_epsilon,
            )
        if residual is not None:
            # TODO: Ideally we want to have (hidden_states+residual)+post_residual_addition.
            # but right now we can only have hidden_states+(residual+post_residual_addition).
            # (hidden_states+residual)+post_residual_addition != hidden_states+(residual+post_residual_addition),
            # we probably need to add another parameter to fused_add_rmsnorm
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            out, _, residual_out = torch_npu.npu_add_rms_norm(
                residual, x, self.weight.data, self.variance_epsilon
            )
            return out, residual_out
        return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]

    def forward_aiter(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            residual_out = torch.empty_like(x)
            output = torch.empty_like(x)
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rms_norm(
                output,
                x,
                residual,
                residual_out,
                self.weight.data,
                self.variance_epsilon,
            )
            return output, residual_out
        return rms_norm(x, self.weight.data, self.variance_epsilon)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.numel() == 0:
            return x
        if self.variance_size_override is not None:
            return self.forward_native(x, residual, post_residual_addition)

        # ── RDNA2 Wave32 HIP kernel (7.78x faster than native) ──
        if _check_rdna2_rmsnorm():
            try:
                if not x.is_contiguous():
                    x = x.contiguous()
                if residual is not None:
                    # Kernel operates in-place: residual += input, input = norm(residual)*w
                    out = x.clone()
                    residual_out = residual.clone()
                    if post_residual_addition is not None:
                        residual_out.add_(post_residual_addition)
                    _rdna2_fused_add_rms_norm_cached(
                        out, residual_out, self.weight.data, self.variance_epsilon
                    )
                    return out, residual_out
                out = torch.empty_like(x)
                _rdna2_rms_norm_cached(out, x, self.weight.data, self.variance_epsilon)
                return out
            except Exception:
                pass

        # ── Existing vllm/aiter chain ──
        if not _has_vllm_rms_norm:
            return self.forward_native(x, residual, post_residual_addition)

        w = self.weight.data + 1.0
        if _use_aiter and not _is_rdna_for_layernorm:
            # aiter API: rms_norm(input, weight, eps) -> output
            #            fused_add_rms_norm(output, input, residual, residual_out, weight, eps)
            if residual is not None:
                output = torch.empty_like(x)
                residual_out = torch.empty_like(x)
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                fused_add_rms_norm(
                    output, x, residual, residual_out, w, self.variance_epsilon
                )
                return output, residual_out
            return rms_norm(x, w, self.variance_epsilon)
        else:
            # vllm or triton fallback API
            if not x.is_contiguous():
                x = x.contiguous()
            if _rms_norm_is_inplace:
                # vllm API: rms_norm(out, input, weight, eps) -> None
                if residual is not None:
                    out = torch.empty_like(x)
                    residual_out = torch.empty_like(x)
                    if post_residual_addition is not None:
                        residual = residual + post_residual_addition
                    fused_add_rms_norm(
                        out, x, residual_out, residual, w, self.variance_epsilon
                    )
                    return out, residual_out
                out = torch.empty_like(x)
                rms_norm(out, x, w, self.variance_epsilon)
                return out
            else:
                # triton fallback: rms_norm(x, w, eps) -> out
                if residual is not None:
                    if post_residual_addition is not None:
                        residual = residual + post_residual_addition
                    x = x + residual
                    out = rms_norm(x, w, self.variance_epsilon)
                    return out, x
                return rms_norm(x, w, self.variance_epsilon)

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if _is_cpu_amx_available:
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                torch.ops.sgl_kernel.gemma_fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.sgl_kernel.gemma_rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        return self.forward_native(x, residual, post_residual_addition)

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if envs.SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM.get():
            return self.forward_native(x, residual)
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            x = x + residual
            residual = x

        x, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
        return x if residual is None else (x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_impl(x, residual, post_residual_addition)

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        use_attn_tp_group: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward with allreduce fusion; uses 1 + weight for fused kernels."""
        # TODO(brayden): we can see if TRTLLM allreduce fusion can provide gemma-style norm
        return _forward_with_allreduce_fusion(
            self,
            x,
            residual,
            post_residual_addition,
            self.weight + 1.0,
            use_attn_tp_group=True,
        )


class Gemma3RMSNorm(MultiPlatformOp):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        # Re-dispatch

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward_native(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def forward_cpu(self, x):
        if _is_cpu_amx_available and x.stride(-1) == 1:
            return torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x, self.weight, self.eps)
        return self.forward_native(x)

    def forward_cuda(self, x):
        return self.forward_native(x)

    def forward_npu(self, x):
        output, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.eps)
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
