# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Optional

import torch
from megatron.core import ModelParallelConfig
from torch import nn
from transformers import LlamaConfig

from verl.utils.megatron_utils import TransformerConfig, convert_config

from .parallel_attention import ParallelLlamaAttention, ParallelLlamaAttentionRmPad
from .parallel_mlp import ParallelLlamaMLP
from .parallel_rmsnorm import ParallelLlamaRMSNorm


class ParallelLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig, layer_idx: int):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = ParallelLlamaAttention(config=config, megatron_config=megatron_config)

        self.mlp = ParallelLlamaMLP(config, megatron_config=megatron_config)
        self.input_layernorm = ParallelLlamaRMSNorm(config, megatron_config)
        self.post_attention_layernorm = ParallelLlamaRMSNorm(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)


        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)


        hidden_states = self.mlp(hidden_states)


        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs


class ParallelLlamaDecoderLayerRmPad(nn.Module):
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig, layer_idx: int):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = ParallelLlamaAttentionRmPad(config=config, megatron_config=megatron_config)

        self.mlp = ParallelLlamaMLP(config, megatron_config=megatron_config)
        self.input_layernorm = ParallelLlamaRMSNorm(config, megatron_config)
        self.post_attention_layernorm = ParallelLlamaRMSNorm(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            sequence_length=sequence_length,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs
