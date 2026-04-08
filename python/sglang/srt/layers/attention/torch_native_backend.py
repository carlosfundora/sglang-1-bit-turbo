from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

    def get_cuda_graph_seq_len_fill_value(self):
        """Fill value used by CUDA graph runners when pre-allocating seq_lens.

        Needed by eagle_draft_cuda_graph_runner even when graphs are disabled,
        as the runner constructor may still read this.  Value 1 matches the
        convention used by triton and flashinfer backends.
        """
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # ── Diagnostic: log supplied vs recomputed prefix at attention entry ──
        try:
            from sglang.srt.observability.spec_decode_tracer import get_tracer, ENABLED
            if ENABLED:
                _diag = get_tracer()
                _sl = seq_lens.cpu().tolist()
                _el = extend_seq_lens.cpu().tolist()
                _pl = extend_prefix_lens.cpu().tolist()
                _recomputed = [max(int(s) - int(e), 0) for s, e in zip(_sl, _el)]
                _diag.log_attention_entry(
                    backend_name="torch_native",
                    forward_batch=type('FB', (), {
                        'forward_mode': 'extend',
                        'batch_size': len(_sl),
                        'seq_lens': seq_lens,
                        'extend_seq_lens': extend_seq_lens,
                        'extend_prefix_lens': extend_prefix_lens,
                    })(),
                    supplied_prefix=_pl,
                    recomputed_prefix=_recomputed,
                )
        except Exception:
            pass
        # ─────────────────────────────────────────────────────────────────────

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]

            # Enforce invariant: prefix + extend == total KV length.
            # extend_seq_len_q is authoritative (matches query token count),
            # so derive prefix from it. Speculative verify batches may have
            # stale prefix/seq_lens that don't sum correctly.
            prefill_seq_len_q = max(int(seq_len_kv) - int(extend_seq_len_q), 0)

            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            if int(extend_seq_len_q) == int(seq_len_kv):
                # Pure extend (no prefix) — skip redundant query trick
                per_req_key = k_cache[
                    req_to_token[req_pool_indices[seq_idx], :seq_len_kv]
                ].movedim(0, query.dim() - 2)
                per_req_value = v_cache[
                    req_to_token[req_pool_indices[seq_idx], :seq_len_kv]
                ].movedim(0, query.dim() - 2)
                if not (per_req_query.dtype == per_req_key.dtype):
                    per_req_key = per_req_key.to(per_req_query.dtype)
                    per_req_value = per_req_value.to(per_req_query.dtype)
                per_req_out = (
                    scaled_dot_product_attention(
                        per_req_query.unsqueeze(0),
                        per_req_key.unsqueeze(0),
                        per_req_value.unsqueeze(0),
                        enable_gqa=enable_gqa,
                        scale=scaling,
                        is_causal=causal,
                    )
                    .squeeze(0)
                    .movedim(query.dim() - 2, 0)
                )
                output[start_q:end_q, :, :] = per_req_out
                start_q, start_kv = end_q, end_kv
                continue

            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
        )

        return o

    def support_triton(self):
        return False


class TorchNativeMultiStepDraftBackend:
    """Wrap multiple TorchNativeAttnBackend instances for EAGLE multi-step
    speculative draft decoding.

    Unlike TritonMultiStepDraftBackend, this does NOT precompute KV indices
    via a triton kernel.  torch_native's per-request loop gathers KV directly
    from ``req_to_token[req_pool_idx, :seq_len_kv]``, so no index buffer is
    needed.  This makes the multi-step wrapper extremely lightweight.

    Required interface (consumed by eagle_worker.py):
        - ``init_forward_metadata(forward_batch)``
        - ``attn_backends``  (list of per-step backends)
        - cuda-graph stubs (no-ops; we run ``--disable-cuda-graph``)
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.device = model_runner.device

        # One backend per draft step (steps - 1 because step 0 uses the
        # target model's own attention backend).
        self.attn_backends: List[TorchNativeAttnBackend] = [
            TorchNativeAttnBackend(model_runner)
            for _ in range(speculative_num_steps - 1)
        ]

        logger.info(
            "TorchNativeMultiStepDraftBackend: topk=%d, steps=%d, backends=%d",
            topk,
            speculative_num_steps,
            len(self.attn_backends),
        )

    # ── Core interface ──────────────────────────────────────────────

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Pre-initialize all per-step backends.

        For torch_native this is a no-op per backend (init_forward_metadata
        is ``pass``), but we still iterate for interface consistency and to
        allow future metadata if needed.
        """
        for backend in self.attn_backends:
            backend.init_forward_metadata(forward_batch)

    # ── CUDA graph stubs (always disabled on ROCm gfx1031) ──────────

    def init_cuda_graph_state(self, *args, **kwargs):
        pass

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        pass

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        pass
