"""
gfxATOM Compatibility Layer for ATOM-RS Integration.

This module provides shim implementations and re-exports for gfxATOM modules
that are referenced by ported attention kernels. It allows kernels to import
from gfxATOM-style module paths while actually using ATOM-RS implementations.

Dependency mapping:
- atom.scheduler -> aiter scheduler (no changes needed for kernels)
- atom.plugins -> feature detection (stub for now)
- atom.quantization -> quantization configs (use torch/aiter native)
- atom.model_config -> model architecture detection (use sglang config)
"""

import torch
import enum
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Scheduler Compatibility (aiter scheduling primitives)
# =============================================================================

class SchedulerPriority(enum.Enum):
    """Priority levels for kernel scheduling."""
    HIGH = 3
    NORMAL = 2
    LOW = 1


class SchedulerConfig:
    """Kernel scheduling configuration."""
    
    def __init__(self):
        self.priority = SchedulerPriority.NORMAL
        self.enable_fusion = True
        self.enable_async = False
        self.max_kernel_size = 65536
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'priority': self.priority.value,
            'enable_fusion': self.enable_fusion,
            'enable_async': self.enable_async,
            'max_kernel_size': self.max_kernel_size,
        }


# =============================================================================
# Plugin System (feature detection & registration)
# =============================================================================

class PluginRegistry:
    """Registry for kernel plugins and feature detection."""
    
    def __init__(self):
        self._plugins: Dict[str, Any] = {}
        self._features: Dict[str, bool] = self._detect_features()
    
    def _detect_features(self) -> Dict[str, bool]:
        """Detect available hardware features."""
        return {
            'has_triton': True,  # Triton is always available
            'has_flash_attention': self._check_flash_attention(),
            'has_fp8': self._check_fp8_support(),
            'has_rocm': self._check_rocm(),
            'has_cuda': torch.cuda.is_available(),
        }
    
    @staticmethod
    def _check_flash_attention() -> bool:
        """Check if FlashAttention is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _check_fp8_support() -> bool:
        """Check if FP8 dtype is supported."""
        try:
            test_tensor = torch.randn(1, dtype=torch.float8_e4m3fn)
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def _check_rocm() -> bool:
        """Check if ROCm is available."""
        try:
            import torch_rocm
            return True
        except ImportError:
            return False
    
    def register_plugin(self, name: str, plugin: Any) -> None:
        """Register a plugin."""
        self._plugins[name] = plugin
        logger.debug(f"Registered plugin: {name}")
    
    def get_plugin(self, name: str) -> Optional[Any]:
        """Get a registered plugin."""
        return self._plugins.get(name)
    
    def has_feature(self, feature: str) -> bool:
        """Check if a feature is available."""
        return self._features.get(feature, False)
    
    def list_features(self) -> Dict[str, bool]:
        """List all available features."""
        return self._features.copy()


# Global plugin registry
_plugin_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _plugin_registry


# =============================================================================
# Quantization Configuration
# =============================================================================

class QuantizationConfig:
    """Quantization configuration for kernels."""
    
    def __init__(
        self,
        use_fp8: bool = True,
        use_int8: bool = False,
        use_int4: bool = False,
        qkv_quantize: bool = True,
        output_quantize: bool = True,
    ):
        self.use_fp8 = use_fp8
        self.use_int8 = use_int8
        self.use_int4 = use_int4
        self.qkv_quantize = qkv_quantize
        self.output_quantize = output_quantize
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfig":
        """Create from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'use_fp8': self.use_fp8,
            'use_int8': self.use_int8,
            'use_int4': self.use_int4,
            'qkv_quantize': self.qkv_quantize,
            'output_quantize': self.output_quantize,
        }


# Default quantization configuration
_default_quant_config = QuantizationConfig()


def get_quantization_config() -> QuantizationConfig:
    """Get the current quantization configuration."""
    return _default_quant_config


def set_quantization_config(config: QuantizationConfig) -> None:
    """Set the quantization configuration."""
    global _default_quant_config
    _default_quant_config = config


# =============================================================================
# Model Configuration & Architecture Detection
# =============================================================================

class AttentionArchitecture(enum.Enum):
    """Attention architecture types supported."""
    MHA = "mha"  # Multi-Head Attention
    GQA = "gqa"  # Grouped Query Attention
    MLA = "mla"  # Multi-head Latent Attention (DeepSeek)
    AITER = "aiter"  # AITER distributed attention
    FLASH = "flash"  # FlashAttention


class ModelArchitectureConfig:
    """Configuration for model architecture and optimization choices."""
    
    def __init__(
        self,
        model_name: str = "",
        attention_arch: AttentionArchitecture = AttentionArchitecture.MHA,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        hidden_dim: int = 4096,
        use_rotary_embedding: bool = True,
        use_fused_kernels: bool = True,
        use_quant: bool = True,
    ):
        self.model_name = model_name
        self.attention_arch = attention_arch
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.hidden_dim = hidden_dim
        self.use_rotary_embedding = use_rotary_embedding
        self.use_fused_kernels = use_fused_kernels
        self.use_quant = use_quant
    
    def is_deepseek_v4(self) -> bool:
        """Check if this is DeepSeek V4 architecture."""
        return "deepseek" in self.model_name.lower() and "v4" in self.model_name.lower()
    
    def is_mla_architecture(self) -> bool:
        """Check if this uses Multi-head Latent Attention."""
        return self.attention_arch == AttentionArchitecture.MLA
    
    def get_attention_backend(self) -> str:
        """Get recommended attention backend."""
        if self.is_deepseek_v4():
            return "deepseek_v4_attn"
        elif self.is_mla_architecture():
            return "aiter_mla"
        else:
            return "triton_mha"


# Global model config
_model_config = ModelArchitectureConfig()


def get_model_config() -> ModelArchitectureConfig:
    """Get the current model architecture configuration."""
    return _model_config


def set_model_config(config: ModelArchitectureConfig) -> None:
    """Set the model architecture configuration."""
    global _model_config
    _model_config = config


def create_model_config_from_hf(model_name: str) -> ModelArchitectureConfig:
    """Create model config from HuggingFace model name."""
    attention_arch = AttentionArchitecture.MHA
    num_heads = 32
    num_kv_heads = 32
    
    if "deepseek" in model_name.lower():
        if "v4" in model_name.lower():
            attention_arch = AttentionArchitecture.MLA
            num_kv_heads = 8  # DeepSeek V4 uses MLA
        else:
            attention_arch = AttentionArchitecture.GQA
    elif "qwen" in model_name.lower():
        attention_arch = AttentionArchitecture.GQA
        num_heads = 32
        num_kv_heads = 8
    elif "llama" in model_name.lower():
        attention_arch = AttentionArchitecture.GQA
        num_heads = 32
        num_kv_heads = 8
    
    return ModelArchitectureConfig(
        model_name=model_name,
        attention_arch=attention_arch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )


# =============================================================================
# Kernel Selector (routes to appropriate backend)
# =============================================================================

class KernelSelector:
    """Selects appropriate kernel implementations based on config."""
    
    def __init__(self):
        self.registry = get_plugin_registry()
        self.model_config = get_model_config()
    
    def get_attention_kernel(self) -> str:
        """Get appropriate attention kernel for current config."""
        backend = self.model_config.get_attention_backend()
        
        if not self.is_backend_available(backend):
            logger.warning(f"Backend {backend} not available, using triton_mha fallback")
            return "triton_mha"
        
        return backend
    
    def get_activation_kernel(self) -> str:
        """Get appropriate activation kernel."""
        quant_config = get_quantization_config()
        if quant_config.use_fp8:
            return "fused_sigmoid_mul_fp8_quant"
        else:
            return "sigmoid_mul"  # reference implementation
    
    def is_backend_available(self, backend: str) -> bool:
        """Check if backend is available."""
        available_backends = {
            "triton_mha": True,  # Always available
            "flash_attn": self.registry.has_feature('has_flash_attention'),
            "deepseek_v4_attn": self.registry.has_feature('has_rocm'),  # Needs ROCm
            "aiter_mla": self.registry.has_feature('has_rocm'),  # Needs ROCm
        }
        return available_backends.get(backend, False)


_kernel_selector = KernelSelector()


def get_kernel_selector() -> KernelSelector:
    """Get the global kernel selector."""
    return _kernel_selector


# =============================================================================
# Exports for gfxATOM-style imports
# =============================================================================

# Make this module compatible with gfxATOM import patterns
__all__ = [
    'SchedulerConfig',
    'SchedulerPriority',
    'PluginRegistry',
    'get_plugin_registry',
    'QuantizationConfig',
    'get_quantization_config',
    'set_quantization_config',
    'ModelArchitectureConfig',
    'AttentionArchitecture',
    'get_model_config',
    'set_model_config',
    'create_model_config_from_hf',
    'KernelSelector',
    'get_kernel_selector',
]

logger.info("✅ gfxATOM compatibility layer initialized")
