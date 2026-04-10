"""GemLite quantization support for SGLang.

GemLite is a pure-Triton low-bit GEMM library that provides optimized
weight-only quantization kernels. Because it uses Triton (not CUDA),
it works on both NVIDIA and AMD ROCm GPUs including RDNA2 (gfx1030).

Supports: A16W4, A16W8, A8W8 with group-wise scales/zeros.

Usage:
  --quantization gemlite              # For models with gemlite quant_config
  --quantization gemlite_awq          # Re-route AWQ models through GemLite Triton
  --quantization gemlite_gptq         # Re-route GPTQ models through GemLite Triton

Reference: https://github.com/mobiusml/gemlite
"""

from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn

from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
)
from sglang.srt.layers.parameter import (
    PackedvLLMParameter,
    GroupQuantScaleParameter,
)
from sglang.srt.utils import is_hip

import logging

logger = logging.getLogger(__name__)

try:
    from gemlite import GemLiteLinear, DType as GemDType, set_autotune
    _GEMLITE_AVAILABLE = True
except ImportError:
    _GEMLITE_AVAILABLE = False
    GemLiteLinear = None
    GemDType = None

_TORCH_TO_GEMLITE_DTYPE = None


def _get_dtype_map():
    global _TORCH_TO_GEMLITE_DTYPE
    if _TORCH_TO_GEMLITE_DTYPE is None and _GEMLITE_AVAILABLE:
        _TORCH_TO_GEMLITE_DTYPE = {
            torch.float16: GemDType.FP16,
            torch.bfloat16: GemDType.BF16,
            torch.float32: GemDType.FP32,
        }
    return _TORCH_TO_GEMLITE_DTYPE or {}


class GemLiteConfig(QuantizationConfig):
    """Config for GemLite pure-Triton weight quantization.

    Reads quantization parameters from model config (HQQ/AWQ/GPTQ format)
    and dispatches to GemLite's Triton GEMM/GEMV kernels.
    """

    def __init__(
        self,
        weight_bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        source_format: str = "gemlite",
        modules_to_not_convert: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if not _GEMLITE_AVAILABLE:
            raise ImportError(
                "GemLite is not installed. Install with: pip install gemlite"
            )
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.source_format = source_format
        self.modules_to_not_convert = modules_to_not_convert or []
        self.pack_factor = 32 // self.weight_bits

        set_autotune("fast")

    def __repr__(self) -> str:
        return (
            f"GemLiteConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"source_format={self.source_format})"
        )

    def get_name(self) -> str:
        return "gemlite"

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # Pure Triton — works on any GPU with Triton support

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GemLiteConfig":
        weight_bits = cls.get_from_keys_or(config, ["w_bit", "bits", "quant_bit"], 4)
        group_size = cls.get_from_keys_or(
            config, ["q_group_size", "group_size"], 128
        )
        zero_point = cls.get_from_keys_or(config, ["zero_point"], True)
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(
            weight_bits=weight_bits,
            group_size=group_size,
            zero_point=zero_point,
            source_format="gemlite",
            modules_to_not_convert=modules_to_not_convert,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["GemLiteLinearMethod"]:
        from sglang.srt.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if any(
                prefix.endswith(m) for m in self.modules_to_not_convert
            ):
                from sglang.srt.layers.quantization.base_config import (
                    UnquantizedLinearMethod,
                )
                return UnquantizedLinearMethod()
            return GemLiteLinearMethod(self)
        return None


class GemLiteAWQConfig(GemLiteConfig):
    """Routes AWQ-format models through GemLite Triton kernels.

    This enables AWQ models on ROCm/AMD GPUs where the original CUDA-only
    AWQ kernels are not available.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("source_format", "awq")
        super().__init__(**kwargs)

    def get_name(self) -> str:
        return "gemlite_awq"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: Dict[str, Any], user_quant: Optional[str]
    ) -> Optional[str]:
        """Claim AWQ models when user specifies gemlite_awq."""
        model_method = hf_quant_cfg.get("quant_method", "").lower()
        if model_method == "awq" and user_quant == "gemlite_awq":
            logger.info(
                "GemLite: intercepting AWQ model → routing through "
                "Triton kernels (gemlite_awq)"
            )
            return "gemlite_awq"
        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GemLiteAWQConfig":
        weight_bits = cls.get_from_keys_or(config, ["w_bit", "bits"], 4)
        group_size = cls.get_from_keys_or(config, ["q_group_size", "group_size"], 128)
        zero_point = cls.get_from_keys_or(config, ["zero_point"], True)
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(
            weight_bits=weight_bits,
            group_size=group_size,
            zero_point=zero_point,
            source_format="awq",
            modules_to_not_convert=modules_to_not_convert,
        )


class GemLiteGPTQConfig(GemLiteConfig):
    """Routes GPTQ-format models through GemLite Triton kernels."""

    def __init__(self, **kwargs):
        kwargs.setdefault("source_format", "gptq")
        super().__init__(**kwargs)

    def get_name(self) -> str:
        return "gemlite_gptq"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GemLiteGPTQConfig":
        weight_bits = cls.get_from_keys_or(config, ["bits"], 4)
        group_size = cls.get_from_keys_or(config, ["group_size"], 128)
        zero_point = cls.get_from_keys_or(config, ["sym"], False)
        return cls(
            weight_bits=weight_bits,
            group_size=group_size,
            zero_point=not zero_point,  # GPTQ: sym=True means no zero_point
            source_format="gptq",
        )


def _unpack_int32_to_uint8(
    packed: torch.Tensor, bits: int, awq_order: bool = True
) -> torch.Tensor:
    """Unpack int32-packed quantized weights to uint8 per-element values.

    AWQ stores multiple low-bit values packed into int32 with an interleaved
    order: [0, 4, 1, 5, 2, 6, 3, 7] for 4-bit. This function unpacks and
    applies the AWQ reverse-order permutation.
    Uses vectorized torch ops — no Python loops.
    """
    elements_per_int32 = 32 // bits
    mask = (1 << bits) - 1
    rows, cols_packed = packed.shape

    # Build shift tensor: [0, bits, 2*bits, ..., (elements-1)*bits]
    shifts = torch.arange(
        0, 32, bits, dtype=torch.int32, device=packed.device
    )  # shape: (elements_per_int32,)

    # Expand packed: (rows, cols_packed, 1) >> (1, 1, elements_per_int32)
    unpacked = (packed.unsqueeze(-1) >> shifts.view(1, 1, -1)) & mask
    # unpacked shape: (rows, cols_packed, elements_per_int32)

    # Flatten to (rows, cols_packed * elements_per_int32)
    unpacked = unpacked.reshape(rows, -1)

    # Apply AWQ reverse-order permutation: [0, 4, 1, 5, 2, 6, 3, 7]
    if bits == 4 and awq_order:
        AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
        reorder_idx = torch.arange(
            unpacked.shape[-1], dtype=torch.int32, device=packed.device
        )
        reorder_idx = reorder_idx.view(-1, elements_per_int32)
        reorder_idx = reorder_idx[:, AWQ_REVERSE_ORDER]
        reorder_idx = reorder_idx.view(-1)
        unpacked = unpacked[:, reorder_idx]

    unpacked = unpacked & mask
    return unpacked.to(torch.uint8)


class GemLiteLinearMethod(LinearMethodBase):
    """Linear method using GemLite's pure-Triton GEMM kernels.

    Stores weights in AWQ/GPTQ int32-packed format during loading,
    then converts to GemLiteLinear in process_weights_after_loading.
    """

    def __init__(self, quant_config: GemLiteConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                f"Input size {input_size_per_partition} not aligned with "
                f"group_size {self.quant_config.group_size}."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        pack_factor = self.quant_config.pack_factor

        # Use PackedvLLMParameter for proper TP sharding of packed int32 weights
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // self.quant_config.group_size
        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        if self.quant_config.zero_point:
            qzeros = PackedvLLMParameter(
                data=torch.empty(
                    num_groups,
                    output_size_per_partition // pack_factor,
                    dtype=torch.int32,
                ),
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=pack_factor,
                weight_loader=weight_loader,
            )
        else:
            qzeros = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.int32),
                requires_grad=False,
            )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        # Stash metadata for process_weights_after_loading
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        if weight_loader is not None:
            # Set weight_loader attrs so the model weight loader can find params
            for name in ["qweight", "scales", "qzeros"]:
                param = getattr(layer, name)
                if hasattr(param, "weight_loader"):
                    continue
                param.weight_loader = weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert loaded int32-packed weights to GemLiteLinear via pack() API.

        AWQ stores: qweight (K, N/pack), scales (n_groups, N), qzeros (n_groups, N/pack)
        GemLite pack() expects: W_q (N, K) uint8, scales (N, n_groups), zeros (N, n_groups)
        """
        device = layer.qweight.data.device
        bits = self.quant_config.weight_bits
        group_size = self.quant_config.group_size
        in_features = layer.input_size_per_partition
        out_features = layer.output_size_per_partition

        scales_dtype = layer.scales.data.dtype
        dtype_map = _get_dtype_map()
        gem_input_dtype = dtype_map.get(scales_dtype, GemDType.FP16)
        gem_output_dtype = gem_input_dtype

        gem_linear = GemLiteLinear(
            W_nbits=bits,
            group_size=group_size,
            in_features=in_features,
            out_features=out_features,
            input_dtype=gem_input_dtype,
            output_dtype=gem_output_dtype,
        )

        # Unpack int32→uint8 on GPU
        W_q = _unpack_int32_to_uint8(layer.qweight.data, bits)
        # AWQ qweight: (K, N_packed) → unpacked (K, N) → transpose to (N, K) for GemLite
        W_q = W_q.t().contiguous()

        # AWQ scales: (n_groups, N) → transpose to (N, n_groups) for pack()
        scales_for_pack = layer.scales.data.t().contiguous()

        # Unpack qzeros and prepare for pack()
        has_zeros = (
            self.quant_config.zero_point and layer.qzeros.data.numel() > 1
        )
        if has_zeros:
            zeros_unpacked = _unpack_int32_to_uint8(layer.qzeros.data, bits)
            # qzeros: (n_groups, N_packed) → unpacked (n_groups, N) → (N, n_groups)
            zeros_for_pack = zeros_unpacked.to(scales_dtype).t().contiguous()
        else:
            zeros_for_pack = None

        # Use GemLite's own pack() — handles repacking, FMA, metadata correctly
        gem_linear.pack(W_q, scales_for_pack, zeros_for_pack)

        layer.gemlite_linear = gem_linear

        del layer.qweight
        del layer.scales
        del layer.qzeros

        logger.info(
            f"GemLite: converted {in_features}x{out_features} "
            f"W{bits} g{group_size} via pack() (group_mode={gem_linear.W_group_mode})"
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gem = layer.gemlite_linear
        out_features = gem.out_features
        out_shape = x.shape[:-1] + (out_features,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        out = gem(reshaped_x)

        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)
