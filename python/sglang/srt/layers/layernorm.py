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

try:
    from sglang.srt.layers.kernels.rdna2.dispatch import rdna2_ops
    from sglang.srt.layers.kernels.rdna2.rmsnorm import (
        fused_add_rms_norm as rdna2_fused_add_rms_norm,
    )
    from sglang.srt.layers.kernels.rdna2.rmsnorm import rms_norm as rdna2_rms_norm
except ImportError:
    pass

import logging
import os
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
        _is_rdna_for_layernorm = (
            _arch.startswith("gfx10")
            or _arch.startswith("gfx11")
            or _arch.startswith("gfx12")
        )
        if _is_rdna_for_layernorm:
            logging.getLogger(__name__).info(
                f"RDNA GPU ({_arch}): AITER CK-based rmsnorm bypassed, using forward_hip chain"
            )
    except Exception:
        pass

# RDNA2 Wave32 HIP kernel — lazy-init (avoids JIT compile at import time)
_rdna2_rmsnorm_checked = False
_rdna2_rmsnorm_ok = False


def _check_rdna2_rmsnorm():
    """Lazy one-time check for RDNA2 RMSNorm kernel availability."""
    global _rdna2_rmsnorm_checked, _rdna2_rmsnorm_ok
    if _rdna2_rmsnorm_checked:
        return _rdna2_rmsnorm_ok
    _rdna2_rmsnorm_checked = True
    if not _is_hip:
        return False
    try:
        _rdna2_rmsnorm_ok = (
            rdna2_ops.probe() and os.environ.get("SGLANG_RDNA2_RMSNORM", "1") != "0"
        )
        if _rdna2_rmsnorm_ok:
            logger.info("RDNA2 Wave32 RMSNorm: enabled for forward_hip dispatch")
    except Exception:
        _rdna2_rmsnorm_ok = False
    return _rdna2_rmsnorm_ok


# TransformerEngine norm — opt-in for MI300+ GPUs without AITER/vLLM
_te_norm_checked = False
_te_norm_available = False
_te_norms_enabled = get_bool_env_var("SGLANG_USE_TE_NORMS") and _is_hip


def _check_te_norm():
    """Lazy check for TransformerEngine norm availability (opt-in via SGLANG_USE_TE_NORMS=1)."""
    global _te_norm_checked, _te_norm_available
    if _te_norm_checked:
        return _te_norm_available
    _te_norm_checked = True
    if not _te_norms_enabled:
        return False
    try:
        import transformer_engine.pytorch as te  # noqa: F401

        _te_norm_available = True
        logger.info("TransformerEngine norms: available (SGLANG_USE_TE_NORMS=1)")
    except ImportError:
        _te_norm_available = False
    return _te_norm_available


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
_has_aiter_layer_norm = False
_has_vllm_rms_norm = False
_rms_norm_is_inplace = False  # True = vllm 4-arg API, False = aiter/triton 3-arg API
if _use_aiter and not _is_rdna_for_layernorm:
    # RDNA GPUs: skip AITER CK-based rmsnorm import — JIT compilation fails on gfx10xx.
    # RDNA path uses forward_hip (RDNA2 HIP kernels) or Triton fallback instead.
    from aiter import layernorm2d_fwd as layer_norm
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm

    _has_aiter_layer_norm = True
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

from sglang.srt.distributed import (
    get_attn_tensor_model_parallel_world_size,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_world_size,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)

try:
    from sglang.srt.layers.flashinfer_comm_fusion import (
        flashinfer_allreduce_residual_rmsnorm,
    )
except ImportError:
    flashinfer_allreduce_residual_rmsnorm = None


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
                if flashinfer_allreduce_residual_rmsnorm is not None:
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
            if (
                _use_aiter
                and not _is_rdna_for_layernorm
                and get_global_server_args().enable_aiter_allreduce_fusion
            ):
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
        # TE adapter: lazily created on first forward_hip call (None=unchecked, False=failed)
        self._te_norm = None

    def _get_te_norm(self):
        """Get or create cached TransformerEngine RMSNorm with shared weight.

        Returns the TE module on success, None on failure. Uses False sentinel
        to avoid retrying after a failed init.
        """
        if self._te_norm is not None:
            return self._te_norm if self._te_norm is not False else None
        if not _check_te_norm() or not self.has_weight:
            self._te_norm = False
            return None
        try:
            import transformer_engine.pytorch as te

            te_mod = te.RMSNorm(self.hidden_size, eps=self.variance_epsilon).to(
                device=self.weight.device, dtype=self.weight.dtype
            )
            # Share weight: TE uses our parameter directly, no duplication.
            # Note: TE's PyTorch forward accesses self.weight at call time (standard
            # nn.Module attribute lookup), so this post-init assignment is safe.
            # If a future TE version caches the pointer at __init__, switch to
            # te.functional.rmsnorm_fwd() which accepts weight as an argument.
            te_mod.weight = self.weight
            self._te_norm = te_mod
            logger.info(
                f"TE RMSNorm adapter created (hidden={self.hidden_size}, "
                f"device={self.weight.device})"
            )
            return self._te_norm
        except Exception as e:
            logger.debug(f"TE RMSNorm init failed: {e}")
            self._te_norm = False
            return None

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
                    rdna2_fused_add_rms_norm(
                        out, residual_out, self.weight.data, self.variance_epsilon
                    )
                    return out, residual_out
                out = torch.empty_like(x)
                rdna2_rms_norm(out, x, self.weight.data, self.variance_epsilon)
                return out
            except Exception as e:
                logger.debug(f"RDNA2 RMSNorm dispatch failed, falling back: {e}")

        # ── Existing vllm/aiter/triton chain ──
        # When TE norms explicitly opted in, try TE first (non-residual only).
        # Otherwise TE falls through to vLLM/aiter/triton which always succeed
        # on functional ROCm installs, making the TE path unreachable.
        if _te_norms_enabled and residual is None:
            te_norm = self._get_te_norm()
            if te_norm is not None:
                try:
                    if not x.is_contiguous():
                        x = x.contiguous()
                    return te_norm(x)
                except Exception as e:
                    logger.debug(f"TE RMSNorm forward failed, falling through: {e}")

        if not _has_vllm_rms_norm:
            return self.forward_native(x, residual, post_residual_addition)

        if not x.is_contiguous():
            x = x.contiguous()

        if _rms_norm_is_inplace:
            # vllm API: rms_norm(out, x, w, eps) -> None (in-place)
            if residual is not None:
                out = torch.empty_like(x)
                residual_out = torch.empty_like(x)
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                fused_add_rms_norm(
                    out,
                    x,
                    residual_out,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                )
                return out, residual_out
            out = torch.empty_like(x)
            rms_norm(out, x, self.weight.data, self.variance_epsilon)
            return out
        else:
            # aiter/triton API: rms_norm(x, w, eps) -> out
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                x = x + residual
                out = rms_norm(x, self.weight.data, self.variance_epsilon)
                return out, x
            return rms_norm(x, self.weight.data, self.variance_epsilon)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = self.override_orig_dtype or x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            if post_residual_addition is not None:
                x = x + post_residual_addition.to(torch.float32)
            if self.fp32_residual:
                residual = x.clone()
            else:
                residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        if self.cast_x_before_out_mul:
            x = self.weight * x.to(orig_dtype)
        else:
            x = (x * self.weight).to(orig_dtype)

        if residual is None:
            return x
        else:
            return x, residual

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
                torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.sgl_kernel.rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        else:
            return self.forward_native(x, residual, post_residual_addition)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual, post_residual_addition)
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        use_attn_tp_group: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward with allreduce fusion, prioritizing flashinfer fused operations."""
        return _forward_with_allreduce_fusion(
            self, x, residual, post_residual_addition, self.weight, use_attn_tp_group
        )


class LayerNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias
        self.dtype = dtype

        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=self.dtype))
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=self.dtype))
        # TE adapter: lazily created on first forward_hip call (None=unchecked, False=failed)
        self._te_norm = None

    def _get_te_norm(self):
        """Get or create cached TransformerEngine LayerNorm with shared weight/bias.

        Returns the TE module on success, None on failure. Uses False sentinel
        to avoid retrying after a failed init.
        """
        if self._te_norm is not None:
            return self._te_norm if self._te_norm is not False else None
        if not _check_te_norm():
            self._te_norm = False
            return None
        try:
            import transformer_engine.pytorch as te

            te_mod = te.LayerNorm(self.hidden_size, eps=self.variance_epsilon).to(
                device=self.weight.device, dtype=self.weight.dtype
            )
            # Share parameters: TE uses ours directly, no duplication.
            # Note: TE's PyTorch forward accesses self.weight at call time (standard
            # nn.Module attribute lookup), so post-init assignment is safe.
            if self.elementwise_affine:
                te_mod.weight = self.weight
            if self.use_bias:
                te_mod.bias = self.bias
            self._te_norm = te_mod
            logger.info(
                f"TE LayerNorm adapter created (hidden={self.hidden_size}, "
                f"device={self.weight.device})"
            )
            return self._te_norm
        except Exception as e:
            logger.debug(f"TE LayerNorm init failed: {e}")
            self._te_norm = False
            return None

    def forward_cuda(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            _flashinfer_layernorm_available
            and x.dtype == torch.bfloat16
            and self.dtype == torch.float32
        ):
            return layernorm(x, self.weight, self.bias, self.variance_epsilon)
        else:
            return self.forward_native(x)

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weight if self.elementwise_affine else None
        bias = self.bias if self.use_bias else None
        orig_dtype = x.dtype
        x = x.to(self.dtype)
        return F.layer_norm(
            x,
            (self.hidden_size,),
            weight=weight,
            bias=bias,
            eps=self.variance_epsilon,
        ).to(orig_dtype)

    def forward_hip(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # TE fused LayerNorm for MI300+ (opt-in via SGLANG_USE_TE_NORMS=1).
        # Checked first when explicitly enabled; falls through on failure.
        if _te_norms_enabled:
            te_norm = self._get_te_norm()
            if te_norm is not None:
                try:
                    if not x.is_contiguous():
                        x = x.contiguous()
                    return te_norm(x)
                except Exception as e:
                    logger.debug(f"TE LayerNorm forward failed, using native: {e}")
        # aiter CK layernorm2d — reduces kernel launches on HIP
        if (
            _has_aiter_layer_norm
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.dtype == self.dtype
        ):
            orig_shape = x.shape
            x = x.reshape(-1, self.hidden_size)
            return layer_norm(x, self.weight, self.bias, self.variance_epsilon).view(
                orig_shape
            )
        # Fast path: avoid casting large input tensors when only weights need
        # dtype alignment.  Cache weight/bias in the input dtype so F.layer_norm
        # runs entirely in bf16/fp16 (~2.6x faster than casting input to f32).
        if x.dtype != self.dtype and x.dtype in (torch.bfloat16, torch.float16):
            if not hasattr(self, "_cached_weight_dtype") or self._cached_weight_dtype != x.dtype:
                self._cached_weight = self.weight.data.to(x.dtype) if self.elementwise_affine else None
                self._cached_bias = self.bias.data.to(x.dtype) if self.use_bias else None
                self._cached_weight_dtype = x.dtype
            return F.layer_norm(
                x, (self.hidden_size,),
                weight=self._cached_weight, bias=self._cached_bias,
                eps=self.variance_epsilon,
            )
        return self.forward_native(x)

    def forward_npu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_cpu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if _is_cpu_amx_available:
            bias_data = self.bias.data if self.use_bias else None
            return torch.ops.sgl_kernel.layernorm_cpu(
                x, self.weight.data, bias_data, self.variance_epsilon
            )
        else:
            return self.forward_native(x)


class GemmaRMSNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def _forward_impl(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            return x, residual
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_impl(x, residual, post_residual_addition)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # ── RDNA2 Wave32 HIP kernel path ──
        if _check_rdna2_rmsnorm():
            try:

                w = self.weight.data + 1.0
                if not x.is_contiguous():
                    x = x.contiguous()
                if residual is not None:
                    out = x.clone()
                    residual_out = residual.clone()
                    if post_residual_addition is not None:
                        residual_out.add_(post_residual_addition)
                    rdna2_fused_add_rms_norm(
                        out, residual_out, w, self.variance_epsilon
                    )
                    return out, residual_out
                out = torch.empty_like(x)
                rdna2_rms_norm(out, x, w, self.variance_epsilon)
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
