from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Optional

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.universal_kv_broker import UniversalKVBroker

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput


class UniversalBrokerAttnBackend(AttentionBackend):
    """
    Attention backend that keeps Triton compute path but inserts UniversalKVBroker
    bookkeeping for hybrid KV modes.
    """

    def __init__(self, model_runner):
        self.runner = model_runner
        self.triton = TritonAttnBackend(model_runner)
        self.broker = UniversalKVBroker(
            gpu_capacity_mb=model_runner.server_args.max_total_tokens // 8
            if model_runner.server_args.max_total_tokens
            else 4096,
            ram_capacity_mb=32768,
        )
        model_tag = int(
            hashlib.sha1(model_runner.server_args.model_path.encode("utf-8")).hexdigest()[:2],
            16,
        )
        self._model_tag = model_tag

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        return self.triton.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        return self.triton.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        return self.triton.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        return self.triton.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.triton.get_cuda_graph_seq_len_fill_value()

    def get_verify_buffers_to_fill_after_draft(self):
        return self.triton.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        return self.triton.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        self._record_block(k, layer=layer, importance=1.0)
        return self.triton.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        self._record_block(k, layer=layer, importance=0.8)
        return self.triton.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
        )

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        self._record_block(k, layer=layer, importance=0.9)
        return self.triton.forward_mixed(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
        )

    def _record_block(self, k: torch.Tensor, layer: RadixAttention, importance: float):
        if self.runner.server_args.kv_cache_dtype not in ("rq3_hybrid", "univ_rq3"):
            return
        layer_id = getattr(layer, "layer_id", 0)
        seq_len = int(k.shape[0]) if k.ndim > 0 else 1
        block_id = self.broker.allocate(self.runner.server_args.model_path, layer_id, seq_len)
        self.broker.compress_and_store(
            block_id,
            k.detach().float(),
            metadata={
                "importance": importance,
                "bit_width": 3,
                "block_size": 16,
                "origin_model_tag": self._model_tag,
            },
        )
