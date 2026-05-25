"""
Shim Modules for gfxATOM Dependencies.

Provides minimal implementations of gfxATOM modules to satisfy imports
in ported kernels without requiring the full gfxATOM infrastructure.

Categories:
1. atom.model_engine.* - model engine, scheduler, operators
2. aiter.* - AITER framework (scheduler, plugins, quantization, distributed)
3. atom.kernel_* - kernel utilities
"""

import sys
import types
from typing import Any, Dict
from sglang.jit_kernel.atom_compat import (
    SchedulerConfig,
    get_plugin_registry,
    get_quantization_config,
    get_model_config,
)


def create_shim_module(name: str, exports: Dict[str, Any]) -> types.ModuleType:
    """Create a shim module with specified exports."""
    module = types.ModuleType(name)
    for key, value in exports.items():
        setattr(module, key, value)
    return module


def create_nested_module(base_name: str, submodules: Dict[str, Dict[str, Any]]) -> types.ModuleType:
    """Create a nested module structure."""
    base = types.ModuleType(base_name)
    for subname, exports in submodules.items():
        full_name = f"{base_name}.{subname}"
        submodule = create_shim_module(full_name, exports)
        setattr(base, subname, submodule)
        sys.modules[full_name] = submodule
    return base


# =============================================================================
# Scheduler Classes
# =============================================================================

class ScheduledBatch:
    """Minimal ScheduledBatch for kernel scheduling."""
    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        self.seq_len = 128
        self.device = "cuda"


class KernelScheduler:
    """Minimal kernel scheduler."""
    def __init__(self):
        self.config = SchedulerConfig()
    
    def schedule(self, kernel_name: str, batch: ScheduledBatch) -> bool:
        return True


# =============================================================================
# Plugin System
# =============================================================================

def get_available_plugins():
    return get_plugin_registry().list_features()

def check_plugin_available(plugin_name: str) -> bool:
    return get_plugin_registry().has_feature(plugin_name)


# =============================================================================
# Build atom.model_engine.* shims
# =============================================================================

atom_model_engine_shims = {
    'scheduler': {
        'ScheduledBatch': ScheduledBatch,
        'KernelScheduler': KernelScheduler,
        'Scheduler': KernelScheduler,
    },
    'operators': {
        'OperatorRegistry': dict,
        'register_operator': lambda name, op: None,
    },
    'config': {
        'ModelConfig': get_model_config().__class__,
        'get_config': get_model_config,
    },
}

# =============================================================================
# Build aiter.* shims
# =============================================================================

aiter_shims = {
    'kernel_scheduler': {
        'ScheduledBatch': ScheduledBatch,
        'KernelScheduler': KernelScheduler,
        'SchedulerConfig': SchedulerConfig,
    },
    'plugins': {
        'get_available': get_available_plugins,
        'has_plugin': check_plugin_available,
        'PluginRegistry': get_plugin_registry().__class__,
    },
    'quantization': {
        'QuantizationConfig': get_quantization_config().__class__,
        'get_config': get_quantization_config,
        'FP8_CONFIG': {'dtype': 'float8_e4m3fn'},
    },
    'model_config': {
        'ModelConfig': get_model_config().__class__,
        'get_config': get_model_config,
    },
    'distributed': {
        'get_rank': lambda: 0,
        'get_world_size': lambda: 1,
        'is_distributed': lambda: False,
        'DistributedConfig': SchedulerConfig,
    },
    'scheduler': {
        'ScheduledBatch': ScheduledBatch,
        'KernelScheduler': KernelScheduler,
    },
}

# =============================================================================
# Build atom.* top-level shims (for "import atom" statements)
# =============================================================================

atom_kernel_utils = {
    'fusion': {
        'enable_fusion': True,
        'fuse_kernels': lambda *args: None,
    },
    'profiling': {
        'enable_profiling': False,
        'profile_kernel': lambda name: None,
    },
}

# =============================================================================
# Install all shims
# =============================================================================

def install_shims():
    """Install all shim modules into sys.modules."""
    
    # Create atom.model_engine.* structure
    for submodule, exports in atom_model_engine_shims.items():
        full_name = f"atom.model_engine.{submodule}"
        sys.modules[full_name] = create_shim_module(full_name, exports)
    
    # Create atom.model_engine base
    atom_model_engine = types.ModuleType("atom.model_engine")
    for submodule in atom_model_engine_shims.keys():
        submod = sys.modules.get(f"atom.model_engine.{submodule}")
        if submod:
            setattr(atom_model_engine, submodule, submod)
    sys.modules["atom.model_engine"] = atom_model_engine
    
    # Create aiter.* structure
    for submodule, exports in aiter_shims.items():
        full_name = f"aiter.{submodule}"
        sys.modules[full_name] = create_shim_module(full_name, exports)
    
    # Create aiter base
    aiter_base = types.ModuleType("aiter")
    for submodule in aiter_shims.keys():
        submod = sys.modules.get(f"aiter.{submodule}")
        if submod:
            setattr(aiter_base, submodule, submod)
    sys.modules["aiter"] = aiter_base
    
    # Create atom.kernel_utils.* structure
    for submodule, exports in atom_kernel_utils.items():
        full_name = f"atom.kernel_utils.{submodule}"
        sys.modules[full_name] = create_shim_module(full_name, exports)
    
    # Create atom.kernel_utils base
    atom_kernel_utils_base = types.ModuleType("atom.kernel_utils")
    for submodule in atom_kernel_utils.keys():
        submod = sys.modules.get(f"atom.kernel_utils.{submodule}")
        if submod:
            setattr(atom_kernel_utils_base, submodule, submod)
    sys.modules["atom.kernel_utils"] = atom_kernel_utils_base
    
    # Create minimal atom base module
    atom_base = types.ModuleType("atom")
    sys.modules["atom"] = atom_base
    
    count = len(sys.modules) - 1  # Rough count
    print(f"✅ Installed shim modules for gfxATOM compatibility")


# Auto-install on import
install_shims()


__all__ = [
    'ScheduledBatch',
    'KernelScheduler',
    'install_shims',
]
