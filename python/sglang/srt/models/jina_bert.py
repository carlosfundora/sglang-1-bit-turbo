# SPDX-License-Identifier: Apache-2.0
# JinaBert (jina-embeddings-v2) for sglang — ALiBi + GEGLU + QK LayerNorm
#
# Reference: jinaai/jina-bert-v2-qk-post-norm (HuggingFace trust_remote_code)
# Architecture: JinaBertForMaskedLM / JinaBertModel
#
# Differences from standard BERT (bert.py):
#   1. ALiBi position encoding instead of absolute position embeddings
#   2. GEGLU feed-forward (gated linear unit with GELU activation)
#   3. Per-head Q/K LayerNorms after projection
#   4. Sandwich-norm residual pattern with extra layer_norm_1 / layer_norm_2

import math
import re
from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

BertConfig = None  # Loaded lazily by sglang from HF config


# ---------------------------------------------------------------------------
# ALiBi slopes — identical to baichuan.py _get_alibi_slopes
# ---------------------------------------------------------------------------

def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


# ---------------------------------------------------------------------------
# GGUF → HF weight name mapping
# ---------------------------------------------------------------------------

_GGUF_TO_HF_PREFIX = {
    "token_embd.weight": "embeddings.word_embeddings.weight",
    "token_embd_norm.weight": "embeddings.LayerNorm.weight",
    "token_embd_norm.bias": "embeddings.LayerNorm.bias",
    "token_types.weight": "embeddings.token_type_embeddings.weight",
}

_GGUF_TO_HF_LAYER = {
    "attn_q.weight": "attention.self.query.weight",
    "attn_q.bias": "attention.self.query.bias",
    "attn_k.weight": "attention.self.key.weight",
    "attn_k.bias": "attention.self.key.bias",
    "attn_v.weight": "attention.self.value.weight",
    "attn_v.bias": "attention.self.value.bias",
    "attn_q_norm.weight": "attention.self.layer_norm_q.weight",
    "attn_q_norm.bias": "attention.self.layer_norm_q.bias",
    "attn_k_norm.weight": "attention.self.layer_norm_k.weight",
    "attn_k_norm.bias": "attention.self.layer_norm_k.bias",
    "attn_output.weight": "attention.output.dense.weight",
    "attn_output.bias": "attention.output.dense.bias",
    "attn_output_norm.weight": "attention.output.LayerNorm.weight",
    "attn_output_norm.bias": "attention.output.LayerNorm.bias",
    "attn_norm_2.weight": "layer_norm_1.weight",
    "attn_norm_2.bias": "layer_norm_1.bias",
    "ffn_up.weight": "mlp.up_gated_layer.weight",
    "ffn_down.weight": "mlp.down_layer.weight",
    "ffn_down.bias": "mlp.down_layer.bias",
    "layer_output_norm.weight": "layer_norm_2.weight",
    "layer_output_norm.bias": "layer_norm_2.bias",
}


def _gguf_to_hf_name(gguf_name: str) -> str:
    """Convert a GGUF tensor name to the HF safetensors equivalent."""
    if gguf_name in _GGUF_TO_HF_PREFIX:
        return _GGUF_TO_HF_PREFIX[gguf_name]

    m = re.match(r"blk\.(\d+)\.(.+)", gguf_name)
    if m:
        layer_idx = m.group(1)
        suffix = m.group(2)
        if suffix in _GGUF_TO_HF_LAYER:
            return f"encoder.layer.{layer_idx}.{_GGUF_TO_HF_LAYER[suffix]}"

    return gguf_name  # passthrough unknown names


# ---------------------------------------------------------------------------
# Embedding (no position embeddings — ALiBi replaces them)
# ---------------------------------------------------------------------------

class JinaBertEmbedding(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.token_type_embeddings = VocabParallelEmbedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        inputs_embeds = self.word_embeddings(input_ids)

        token_type_ids = forward_batch.token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_ids.shape, dtype=torch.long, device=inputs_embeds.device
            )

        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeddings = self.LayerNorm(inputs_embeds + token_type_embeds)
        return embeddings


# ---------------------------------------------------------------------------
# Self-attention with ALiBi + QK LayerNorm (bypasses RadixAttention)
# ---------------------------------------------------------------------------

class JinaBertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        layer_norm_eps: float,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # QK post-norm (Jina's signature)
        self.layer_norm_q = nn.LayerNorm(self.q_size, eps=layer_norm_eps)
        self.layer_norm_k = nn.LayerNorm(self.kv_size, eps=layer_norm_eps)

        # ALiBi slopes for this TP shard
        tp_rank = get_tensor_model_parallel_rank()
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        all_slopes = _get_alibi_slopes(self.total_num_heads)
        self.register_buffer(
            "alibi_slopes",
            all_slopes[head_start:head_end].float(),
            persistent=False,
        )

        self._alibi_cache_size = 0
        self._alibi_cache: Optional[torch.Tensor] = None

    def _get_alibi_bias(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Compute symmetric ALiBi bias: -slope * |i - j|.
        Returns shape [1, num_heads, seq_len, seq_len]."""
        if self._alibi_cache is not None and self._alibi_cache_size >= seq_len:
            return self._alibi_cache[:, :, :seq_len, :seq_len].to(dtype)

        positions = torch.arange(seq_len, device=device)
        distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        # [num_heads, seq_len, seq_len]
        bias = -self.alibi_slopes.unsqueeze(1).unsqueeze(2) * distance.unsqueeze(0).float()
        bias = bias.unsqueeze(0)  # [1, num_heads, seq_len, seq_len]

        self._alibi_cache = bias
        self._alibi_cache_size = seq_len
        return bias.to(dtype)

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK post-norm
        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)

        # Process each sequence in the packed batch
        seq_lens = forward_batch.extend_seq_lens
        outputs = []
        offset = 0

        for length in seq_lens:
            seq_len = int(length)
            qi = q[offset : offset + seq_len]
            ki = k[offset : offset + seq_len]
            vi = v[offset : offset + seq_len]

            # [1, num_heads, seq_len, head_dim]
            qi = qi.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            ki = ki.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            vi = vi.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            alibi_bias = self._get_alibi_bias(seq_len, qi.device, qi.dtype)

            attn_out = F.scaled_dot_product_attention(
                qi, ki, vi, attn_mask=alibi_bias, is_causal=False
            )
            # [1, num_heads, seq_len, head_dim] → [seq_len, num_heads * head_dim]
            attn_out = attn_out.transpose(1, 2).reshape(seq_len, -1)
            outputs.append(attn_out)
            offset += seq_len

        return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Attention output (dense + residual + LayerNorm)
# ---------------------------------------------------------------------------

class JinaBertSelfOutput(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_norm_eps: float,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dense = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# ---------------------------------------------------------------------------
# Attention block (self-attention + output)
# ---------------------------------------------------------------------------

class JinaBertAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        layer_norm_eps: float,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = JinaBertSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            layer_norm_eps=layer_norm_eps,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.output",
        )
        self.output = JinaBertSelfOutput(
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.output",
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        self_output = self.self_attn(hidden_states, forward_batch)
        return self.output(self_output, hidden_states)


# ---------------------------------------------------------------------------
# GEGLU MLP
# ---------------------------------------------------------------------------

class JinaBertGEGLUMLP(nn.Module):
    """GEGLU feed-forward: up * gelu(gate), then down-project.

    Jina packs [up | gate] into a single weight of shape [2*intermediate, hidden].
    The first half is the value (up), the second half gets the GELU gate.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.intermediate_size = intermediate_size

        self.up_gated_layer = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_gated_layer",
        )
        self.down_layer = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_layer",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.up_gated_layer(hidden_states)
        # Jina convention: first half = up (value), second half = gate
        d = hidden_states.shape[-1] // 2
        up = hidden_states[..., :d]
        gate = hidden_states[..., d:]
        hidden_states = up * F.gelu(gate)
        hidden_states, _ = self.down_layer(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Transformer layer (sandwich-norm)
# ---------------------------------------------------------------------------

class JinaBertLayer(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.attention = JinaBertAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        self.mlp = JinaBertGEGLUMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.layer_norm_1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layer_norm_2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        residual = hidden_states
        attn_output = self.attention(hidden_states, forward_batch)
        residual = self.layer_norm_1(residual + attn_output)
        mlp_output = self.mlp(residual)
        output = self.layer_norm_2(residual + mlp_output)
        return output


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class JinaBertEncoder(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                JinaBertLayer(
                    config=config,
                    layer_id=layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layer.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states, forward_batch)
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class JinaBertModel(nn.Module):
    def __init__(
        self,
        *,
        config: BertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embeddings = JinaBertEmbedding(config)
        self.encoder = JinaBertEncoder(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )
        self.pooler = Pooler(pooling_type=PoolingType.MEAN, normalize=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        assert get_embedding is True

        hidden_states = self.embeddings(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
        )
        hidden_states = self.encoder(hidden_states, forward_batch=forward_batch)
        hidden_states = self.pooler(hidden_states, forward_batch)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "query", "q"),
            ("qkv_proj", "key", "k"),
            ("qkv_proj", "value", "v"),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # GGUF name conversion
            if name.startswith("blk.") or name.startswith("token_"):
                name = _gguf_to_hf_name(name)

            # HF → sglang rename: attention.self → attention.self_attn
            name = name.replace("attention.self.", "attention.self_attn.")

            # Skip MLM head weights (not used for embedding)
            if name.startswith("cls.") or name.startswith("predictions."):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    break
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


class JinaBertForMaskedLM(JinaBertModel):
    """Thin alias so sglang auto-detects the HF architecture name."""
    pass


EntryClass = [JinaBertModel, JinaBertForMaskedLM]
