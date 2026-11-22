#!/usr/bin/env python3
"""
Hybrid AITER + Triton Backend for ATOM
========================================

Enables BOTH AITER and Triton kernels to run concurrently:
- AITER handles: decode attention (Wave32-optimized, gfx1030 tuned)  
- Triton handles: prefill attention, fallback ops, experimental layers
- Dynamic dispatch based on: shape, sequence length, batch size

This enables **best-of-breed** kernel selection per operation instead of
falling back to a single backend.

Key insight: AITER and Triton are complementary:
- AITER: Small batch, decoding, Wave32-friendly, ROCm-native
- Triton: Large batch, prefill, flexible shapes, cross-platform
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HybridAttnProfile:
    """Profile for hybrid AITER+Triton dispatch decisions"""
    
    # AITER (AMD Wave32) preferred conditions
    aiter_preferred_batch_min: int = 1
    aiter_preferred_batch_max: int = 32
    aiter_preferred_seq_min: int = 1
    aiter_preferred_seq_max: int = 16384  # gfx1030 L2 friendly
    
    # Triton preferred conditions
    triton_preferred_batch_min: int = 32
    triton_preferred_batch_max: int = 512
    triton_preferred_seq_min: int = 128
    triton_preferred_seq_max: int = 131072
    
    # Fallback strategy
    fallback_primary: str = "triton"  # Use Triton if both unavailable
    

class HybridAITERTritonBackend:
    """
    Dual-dispatch attention backend combining AITER and Triton.
    
    Dispatch logic:
    1. Decode phase (small batch, single token): → AITER (Wave32-optimized)
    2. Prefill phase (large batch, long sequence): → Triton (flexible)
    3. Out-of-profile shapes: → Fallback (Triton by default)
    4. Both unavailable: → torch.nn.functional.scaled_dot_product_attention
    """
    
    def __init__(self, runner, profile: Optional[HybridAttnProfile] = None):
        self.runner = runner
        self.profile = profile or HybridAttnProfile()
        self.aiter_backend = None
        self.triton_backend = None
        self.stats = {
            "aiter_calls": 0,
            "triton_calls": 0,
            "fallback_calls": 0,
            "dispatch_decisions": [],
        }
        
        # Initialize both backends if available
        self._init_backends()
    
    def _init_backends(self):
        """Initialize AITER and Triton backends if available"""
        try:
            from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
            from sglang.srt.server_args import _is_hip_aiter_available
            
            if _is_hip_aiter_available():
                self.aiter_backend = AiterAttnBackend(self.runner)
                logger.info("✓ AITER backend initialized")
            else:
                logger.warning("⊘ AITER unavailable (HIP/gfx1030 not detected)")
        except Exception as e:
            logger.warning(f"⊘ AITER init failed: {e}")
        
        try:
            from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
            
            self.triton_backend = TritonAttnBackend(self.runner)
            logger.info("✓ Triton backend initialized")
        except Exception as e:
            logger.warning(f"⊘ Triton init failed: {e}")
    
    def select_backend(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        is_prefill: bool = False,
    ) -> str:
        """
        Select optimal backend for given shapes.
        
        Returns: "aiter", "triton", or "fallback"
        """
        
        # Prefill → prefer Triton (better for large batches)
        if is_prefill and self.triton_backend:
            if (self.profile.triton_preferred_batch_min <= batch_size <= self.profile.triton_preferred_batch_max and
                self.profile.triton_preferred_seq_min <= seq_len <= self.profile.triton_preferred_seq_max):
                return "triton"
        
        # Decode → prefer AITER (Wave32-optimized)
        if not is_prefill and self.aiter_backend:
            if (self.profile.aiter_preferred_batch_min <= batch_size <= self.profile.aiter_preferred_batch_max and
                self.profile.aiter_preferred_seq_min <= seq_len <= self.profile.aiter_preferred_seq_max):
                return "aiter"
        
        # Shape-based dispatch
        if batch_size <= 16 and self.aiter_backend:
            return "aiter"  # Small batch → AITER (gfx1030 efficient)
        
        if batch_size > 16 and self.triton_backend:
            return "triton"  # Large batch → Triton (parallel-friendly)
        
        # Fallback
        return self.profile.fallback_primary
    
    def forward(
        self,
        query,
        key,
        value,
        causal_mask=None,
        kv_seq_len: Optional[int] = None,
        is_prefill: bool = False,
        **kwargs
    ):
        """
        Hybrid forward pass with automatic backend dispatch.
        """
        
        batch_size = query.shape[0] if len(query.shape) > 2 else 1
        seq_len = query.shape[1] if len(query.shape) > 2 else query.shape[0]
        num_heads = query.shape[-2] if len(query.shape) > 2 else 1
        head_dim = query.shape[-1]
        
        # Select backend
        selected = self.select_backend(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            is_prefill=is_prefill,
        )
        
        # Record dispatch decision
        self.stats["dispatch_decisions"].append({
            "batch": batch_size,
            "seq_len": seq_len,
            "backend": selected,
            "is_prefill": is_prefill,
        })
        
        # Invoke appropriate backend
        try:
            if selected == "aiter" and self.aiter_backend:
                self.stats["aiter_calls"] += 1
                return self.aiter_backend.forward(
                    query, key, value, causal_mask=causal_mask, 
                    kv_seq_len=kv_seq_len, **kwargs
                )
            
            elif selected == "triton" and self.triton_backend:
                self.stats["triton_calls"] += 1
                return self.triton_backend.forward(
                    query, key, value, causal_mask=causal_mask,
                    kv_seq_len=kv_seq_len, **kwargs
                )
            
            else:
                # Fallback to torch
                self.stats["fallback_calls"] += 1
                return self._torch_fallback(query, key, value, causal_mask)
        
        except Exception as e:
            logger.warning(f"Backend {selected} failed: {e}. Falling back.")
            self.stats["fallback_calls"] += 1
            return self._torch_fallback(query, key, value, causal_mask)
    
    def _torch_fallback(self, query, key, value, causal_mask):
        """Fallback to torch.nn.functional.scaled_dot_product_attention"""
        import torch
        import torch.nn.functional as F
        
        # Scale query
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        
        # Apply causal mask if needed
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        return torch.matmul(attn_weights, value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return dispatch statistics for analysis"""
        return {
            "aiter_kernel_calls": self.stats["aiter_calls"],
            "triton_kernel_calls": self.stats["triton_calls"],
            "fallback_calls": self.stats["fallback_calls"],
            "total_calls": sum([
                self.stats["aiter_calls"],
                self.stats["triton_calls"],
                self.stats["fallback_calls"],
            ]),
            "aiter_usage_percent": (
                100.0 * self.stats["aiter_calls"] / 
                max(1, sum([
                    self.stats["aiter_calls"],
                    self.stats["triton_calls"],
                    self.stats["fallback_calls"],
                ]))
            ),
        }


def create_hybrid_atom_backend(runner):
    """
    Factory function for registering hybrid ATOM backend in SGLang.
    
    Usage in attention_registry.py:
        @register_attention_backend("atom_hybrid")
        def create_hybrid_atom(runner):
            return create_hybrid_atom_backend(runner)
    
    Then launch with:
        python3 -m sglang.launch_server --model-path=... --attention-backend=atom_hybrid
    """
    return HybridAITERTritonBackend(runner)


if __name__ == "__main__":
    # Demo: Show dispatch decisions
    profile = HybridAttnProfile()
    
    print("\n" + "="*70)
    print("HYBRID AITER + TRITON BACKEND DISPATCH DECISIONS")
    print("="*70 + "\n")
    
    test_cases = [
        (1, 1, 32, 64, False, "Single token decode"),
        (1, 512, 32, 64, True, "Prefill 512 tokens"),
        (4, 256, 32, 64, False, "Batch decode 256 tokens"),
        (16, 1024, 32, 64, True, "Batch prefill 1024 tokens"),
        (64, 4096, 32, 64, True, "Large batch prefill"),
        (128, 16384, 32, 64, False, "Large batch long context"),
    ]
    
    class MockRunner:
        pass
    
    backend = HybridAITERTritonBackend(MockRunner())
    
    for batch, seq, heads, dim, prefill, desc in test_cases:
        selected = backend.select_backend(batch, seq, heads, dim, is_prefill=prefill)
        phase = "PREFILL" if prefill else "DECODE "
        print(f"[{phase}] B={batch:3d} S={seq:5d} → {selected:7s}  ({desc})")
    
    print("\n" + "="*70)
