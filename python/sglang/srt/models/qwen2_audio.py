# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/1d45d90e5d1552eccb6d8cc9b7bba283ccefb808/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Qwen2AudioEncoderConfig, Qwen2Config
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioEncoder,
    Qwen2AudioMultiModalProjector,
)

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def _concat_audio_features_with_preallocation(
    items: List[MultimodalDataItem], dtype: torch.dtype
) -> torch.Tensor:
    """Concatenate item.feature tensors with a single preallocated output buffer."""
    first_feature = items[0].feature
    trailing_shape = first_feature.shape[1:]
    if any(item.feature.shape[1:] != trailing_shape for item in items):
        # Fall back when feature ranks/shapes differ across items.
        return torch.cat([item.feature for item in items], dim=0).to(dtype=dtype)

    total_rows = sum(int(item.feature.shape[0]) for item in items)
    output = first_feature.new_empty(
        (total_rows, *trailing_shape), dtype=dtype, device=first_feature.device
    )

    offset = 0
    for item in items:
        feature = item.feature
        rows = int(feature.shape[0])
        output[offset : offset + rows].copy_(feature.to(dtype=dtype))
        offset += rows
    return output


def _concat_audio_feature_lens_with_preallocation(
    items: List[MultimodalDataItem],
) -> torch.Tensor:
    """Concatenate item.audio_feature_lens with one preallocated vector."""
    first_lens = items[0].audio_feature_lens
    total = sum(int(item.audio_feature_lens.numel()) for item in items)
    output = first_lens.new_empty((total,), dtype=first_lens.dtype, device=first_lens.device)

    offset = 0
    for item in items:
        lens = item.audio_feature_lens.reshape(-1)
        n = int(lens.numel())
        output[offset : offset + n].copy_(lens)
        offset += n
    return output


def _flatten_projected_audio_embeds_with_preallocation(
    audio_embeds: torch.Tensor, audio_feature_lens: torch.Tensor
) -> torch.Tensor:
    """Pack variable-length projected audio embeddings into one contiguous tensor."""
    hidden_size = int(audio_embeds.shape[-1])
    total_tokens = int(audio_feature_lens.sum().item())
    output = audio_embeds.new_empty((total_tokens, hidden_size))

    offset = 0
    for feature_len, embed in zip(audio_feature_lens, audio_embeds):
        token_count = int(feature_len.item())
        if token_count <= 0:
            continue
        end = offset + token_count
        output[offset:end].copy_(embed[:token_count])
        offset = end
    return output[:offset]


class Qwen2AudioForConditionalGeneration(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen2AudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if getattr(self.config, "audio_config", None) is None:
            self.config.audio_config = Qwen2AudioEncoderConfig(
                self.config._name_or_path
            )

        if getattr(self.config, "text_config", None) is None:
            self.config.text_config = Qwen2Config(self.config._name_or_path)

        self.audio_tower = Qwen2AudioEncoder(
            config.audio_config,
        )
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(config)
        self.language_model = Qwen2ForCausalLM(
            config.text_config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        input_features = _concat_audio_features_with_preallocation(
            items, self.audio_tower.dtype
        )

        audio_embeds = self.audio_tower(input_features).last_hidden_state
        audio_embeds = self.multi_modal_projector(audio_embeds)

        audio_feature_lens = _concat_audio_feature_lens_with_preallocation(items)
        return _flatten_projected_audio_embeds_with_preallocation(
            audio_embeds, audio_feature_lens
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "audio_tower" in name:
                    continue
                name_tmp = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name_tmp.endswith(".bias") and name_tmp not in params_dict:
                    continue
                param = params_dict[name_tmp]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = Qwen2AudioForConditionalGeneration
