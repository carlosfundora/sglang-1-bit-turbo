"""RDNA2 (gfx1030) HIP decode-attention backend.

Inherits everything from TritonAttnBackend (metadata, prefill/extend, cuda-graph)
and overrides ONLY forward_decode to use the native HIP paged flash-decode kernel
(`rdna2_hip_decode.paged_decode`, the device-resident split-K kernel that mirrors
the canonical rs_rdna2_kernels::flash_decode). Falls back to the triton decode for
any case the HIP kernel doesn't cover (MLA, sliding window, logit softcap, quantized
KV, non-fp16, or odd head_dim) so it is always safe to select.

STATUS (measured 2026-06-14, RX 6700 XT, Qwen2.5-0.5B f16): EXPERIMENTAL — correct
but NOT yet faster than triton. d512 ~52 vs triton ~57 (decode is matmul-bound there);
d2048/d4096 it DEGRADES (37/27 vs triton's flat ~57) because the kernel only splits KV
across <=4 warps within ONE block, so long contexts serialize. triton/llama split KV
across many BLOCKS that scale with seq_k. TODO to win: inter-block split-K (grid.y =
parallel_blocks(seq_k) + a combine pass), the same design as llama.cpp fattn-vec.
KEY FINDING: the llama.cpp (181) vs sglang (~57) decode gap on small models is sglang's
per-token FRAMEWORK overhead, not the attention kernel (all sglang backends cluster at
52-60 regardless of attention kernel) — so swapping the attention kernel alone can't
close it. Keep triton/universal_broker as the default RDNA2 backends.
"""

import logging

import torch

from sglang.srt.layers.attention import rdna2_hip_decode
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

logger = logging.getLogger(__name__)


class Rdna2HipAttnBackend(TritonAttnBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hip_decode_ok = rdna2_hip_decode.is_available()
        if not self._hip_decode_ok:
            logger.warning(
                "rdna2_hip backend: HIP paged-decode kernel unavailable; "
                "decode will fall back to the triton path."
            )

    def _hip_decode_supported(self, layer, q) -> bool:
        if not self._hip_decode_ok:
            return False
        if getattr(self, "use_mla", False):
            return False
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            return False
        if getattr(layer, "logit_cap", 0) not in (0, 0.0, None):
            return False
        if layer.k_scale is not None or layer.v_scale is not None:
            return False  # quantized KV -> needs descale; use triton
        if q.dtype != torch.float16:
            return False
        if layer.qk_head_dim != layer.v_head_dim:
            return False
        if layer.qk_head_dim % 32 != 0 or layer.qk_head_dim > 256:
            return False
        if layer.tp_q_head_num % layer.tp_k_head_num != 0:
            return False
        return True

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        if not self._hip_decode_supported(layer, q):
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )

        Hq = layer.tp_q_head_num
        Hk = layer.tp_k_head_num
        D = layer.qk_head_dim
        q = q.reshape(-1, Hq * D)
        o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices
        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        try:
            rdna2_hip_decode.paged_decode(
                q.view(-1, Hq, D),
                k_buffer.view(-1, Hk, D),
                v_buffer.view(-1, Hk, D),
                o.view(-1, Hq, D),
                kv_indptr,
                kv_indices,
                layer.scaling,
            )
        except Exception as e:  # safety: never hard-fail a decode
            logger.warning(f"rdna2_hip decode fell back to triton: {e}")
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=False, **kwargs
            )
        return o
