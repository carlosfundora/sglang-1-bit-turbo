"""
ATOM Backend Registration and Dispatch Integration for SGLang

This module registers ATOM as a first-class attention backend in SGLang's
attention registry, enabling:

1. Selection via --attention-backend atom
2. Automatic hardware detection and fallback
3. Seamless integration with existing SGLang infrastructure
4. Non-breaking optional activation
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def register_atom_backend():
    """
    Register ATOM backend with SGLang attention registry.
    
    This should be called once at server startup in launch_server.py.
    
    After registration:
    - Users can specify --attention-backend atom
    - SGLang will automatically detect hardware capability
    - Fallback to Triton/AITER if ATOM unavailable
    """
    try:
        from sglang.srt.layers.attention.attention_registry import (
            get_attention_registry,
        )
        from sglang.srt.layers.attention.atom_backend import ATOMBackend
        
        registry = get_attention_registry()
        
        # Register ATOM backend
        registry.register("atom", ATOMBackend)
        
        logger.info("[ATOMRegistration] ✓ ATOM backend registered successfully")
        return True
    
    except ImportError as e:
        logger.warning(f"[ATOMRegistration] Failed to import SGLang modules: {e}")
        return False
    except Exception as e:
        logger.warning(f"[ATOMRegistration] Error registering ATOM backend: {e}")
        return False


def get_atom_backend_info() -> dict:
    """
    Get information about ATOM backend availability and capabilities.
    
    Returns:
        {
            "available": bool,
            "hardware_mode": str,
            "capabilities": [str],
            "supported_dtypes": [str],
        }
    """
    try:
        from sglang.srt.layers.attention.atom_backend import (
            ATOMBackend,
            ATOMHardwareMode,
        )
        import torch
        
        # Create temporary instance to check capabilities
        backend = ATOMBackend()
        
        return {
            "available": backend.atom_available,
            "hardware_mode": str(backend.hardware_mode),
            "enable_wave32": backend.enable_wave32,
            "enable_fp8_prefill": backend.enable_fp8_prefill,
            "enable_quantized_kv": backend.enable_quantized_kv,
            "fallback_to_triton": backend.fallback_to_triton,
            "capabilities": [
                "decode_attention",
                "prefill_attention",
                "speculative_decoding",
                "quantized_kv",
                "fused_rmnorm_rope",
            ] if backend.atom_available else [],
            "supported_dtypes": ["fp16", "bf16", "fp8"] if backend.atom_available else [],
        }
    
    except Exception as e:
        logger.warning(f"[ATOMRegistration] Error getting backend info: {e}")
        return {
            "available": False,
            "hardware_mode": "UNAVAILABLE",
            "capabilities": [],
            "supported_dtypes": [],
        }


def setup_atom_backend_with_flags(args) -> Optional[str]:
    """
    Configure ATOM backend based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments from launch_server
    
    Returns:
        Selected attention backend name, or None if ATOM not suitable
    
    Usage in launch_server.py:
        backend = setup_atom_backend_with_flags(args)
        if backend == "atom":
            engine = create_engine(attention_backend="atom", ...)
    """
    if not hasattr(args, "attention_backend"):
        return None
    
    if args.attention_backend != "atom":
        return None
    
    try:
        from sglang.srt.layers.attention.atom_backend import ATOMBackend
        
        backend = ATOMBackend()
        
        if not backend.atom_available:
            logger.warning(
                "[ATOMSetup] ATOM not available on this system. "
                "Falling back to default backend."
            )
            return None
        
        logger.info(
            f"[ATOMSetup] ✓ Using ATOM backend "
            f"(hardware_mode={backend.hardware_mode})"
        )
        
        return "atom"
    
    except Exception as e:
        logger.warning(f"[ATOMSetup] Error setting up ATOM backend: {e}")
        return None


# ============================================================================
# INTEGRATION PATCHES FOR SGLang
# ============================================================================

"""
To integrate ATOM backend into SGLang, apply these patches:

PATCH 1: In python/sglang/launch_server.py
After line where arguments are parsed, add:

    from sglang.srt.layers.attention.atom_backend_registry import (
        register_atom_backend,
        setup_atom_backend_with_flags,
    )
    
    # Register ATOM backend
    register_atom_backend()
    
    # Configure based on flags
    attention_backend = setup_atom_backend_with_flags(args)
    if attention_backend:
        args.attention_backend = attention_backend


PATCH 2: In python/sglang/server_args.py
Add argument to parser:

    parser.add_argument(
        "--attention-backend",
        type=str,
        default="auto",
        choices=["auto", "triton", "aiter", "atom", "flashinfer", "etc."],
        help="Attention backend to use. 'auto' picks best available. 'atom' uses gfxATOM (AMD).",
    )
    
    parser.add_argument(
        "--atom-wave32",
        type=bool,
        default=True,
        help="Use wave32 execution for ATOM on RDNA2 (faster for small batches).",
    )
    
    parser.add_argument(
        "--atom-fallback-to-triton",
        type=bool,
        default=True,
        help="Fallback to Triton if ATOM kernel fails.",
    )


PATCH 3: In python/sglang/srt/layers/attention/attention_registry.py
Modify get_attention_backend() to support ATOM:

    def get_attention_backend(backend_name: str, **kwargs):
        if backend_name == "atom":
            from sglang.srt.layers.attention.atom_backend import ATOMBackend
            return ATOMBackend(**kwargs)
        # ... existing backends ...


PATCH 4: In python/sglang/launch_server.py (model executor initialization)
Pass ATOM flags to backend:

    attention_backend_kwargs = {
        "enable_wave32": args.atom_wave32,
        "fallback_to_triton": args.atom_fallback_to_triton,
    }
    
    backend = get_attention_backend(
        args.attention_backend,
        **attention_backend_kwargs
    )
"""

# ============================================================================
# DIRECT PAGED KV ACCESS GUARANTEE
# ============================================================================

"""
ATOM Backend Access to SGLang Paged KV Pools

The ATOMBackend._extract_paged_kv_metadata() method ensures:

1. ZERO COPY - Direct tensor pointers to GPU memory
   → req_to_token_pool: [num_reqs, max_context_len] int64
   → token_to_kv_pool: [num_pages * page_size, 2, hidden_dim] fp16
   → page_table: [num_reqs, max_pages] int32
   → kv_indptr: [num_reqs + 1] int32
   → kv_indices: [total_kv_tokens] int32

2. NO RESHAPE - Tensors already in optimal layout for HIP kernels

3. ATOMIC ACCESS - All pointers point to stable SGLang-managed buffers
   → Allocation happens in RadixCache.init()
   → Lifetime = entire forward pass
   → Safe concurrent reads by ATOM kernels

4. QUANTIZATION TRANSPARENT - Optional compression layers
   → If RotorQuant enabled: compression_mode="rq4_planar"
   → If TurboQuant enabled: compression_mode="tq3"
   → ATOM kernels handle decompression internally

5. HARDWARE EFFICIENT - Paged layout matches AMD GPU expectations
   → Page-strided access patterns
   → Wave64/Wave32 occupancy friendly
   → PCIe-free (direct GPU reads)
"""
