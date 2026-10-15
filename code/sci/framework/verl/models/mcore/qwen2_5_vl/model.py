# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Alibaba PAI Team.
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
import logging

import torch
from megatron.core import InferenceParams, mpu, tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from verl.models.mcore.util import preprocess_packed_seqs

from .attention import Qwen2_5VLSelfAttention
from .vision_model import Qwen2_5VisionModel


class Qwen2_5VLModel(MegatronModule):

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        parallel_output: bool = True,
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 10000,
        fp16_lm_cross_entropy: bool = False,
        language_share_embeddings_and_output_weights: bool = False,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
    ) -> None:
        super().__init__(config=language_transformer_config)

        vision_transformer_layer_spec.submodules.self_attention.module = Qwen2_5VLSelfAttention
        for layer_spec in language_transformer_layer_spec.layer_specs:
            layer_spec.submodules.self_attention.module = Qwen2_5VLSelfAttention

        logging.getLogger(__name__).warning("Qwen2VL model is under development and may be missing features.")

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        self.square_merge_size = vision_projection_config.ffn_hidden_size // vision_transformer_config.hidden_size

        self.share_embeddings_and_output_weights = False
        if self.pre_process:
            self.vision_model = Qwen2_5VisionModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                vision_projection_config,
                vision_projection_layer_spec,
                projection_type=vision_projection_type,
                pre_process=True,
                post_process=True,
            )

        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_rotary_base,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
        )
        assert mpu.get_context_parallel_world_size() <= 1, "please use mbridge for qwen2_5_vl with context parallelism"
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def shared_embedding_or_output_weight(self):
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen2VL"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool):
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        video_start_index = 0
        vision_grid_thw = None
        vision_data = None
        if image_grid_thw is not None:
            image_mask = input_ids == self.image_token_id
            vision_grid_thw = image_grid_thw
            vision_data = pixel_values
            video_start_index = image_mask.sum().item()
        if video_grid_thw is not None:
            video_mask = input_ids == self.video_token_id
            if vision_grid_thw is not None:
                vision_grid_thw = torch.cat([vision_grid_thw, video_grid_thw], dim=0)
                vision_data = torch.cat([vision_data, pixel_values_videos], dim=0)
            else:
                vision_grid_thw = video_grid_thw
                vision_data = pixel_values_videos
        use_inference_kv_cache = (
            inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if use_inference_kv_cache:
            raise NotImplementedError()

        if self.pre_process:
            vision_embeds = None
            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                vision_embeds = self.vision_model(
                    vision_data=vision_data,
                    grid_thw=vision_grid_thw,
                )

            if inference_params is not None:
                raise NotImplementedError()

            if use_inference_kv_cache:
                language_embeddings: torch.Tensor = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None,
                )
                combined_embeddings = language_embeddings
            elif vision_embeds is not None:
                if video_start_index == 0:
                    image_embeds = None
                    video_embeds = vision_embeds
                elif video_start_index == vision_embeds.shape[0]:
                    image_embeds = vision_embeds
                    video_embeds = None
                elif 0 < video_start_index < vision_embeds.shape[0]:
                    image_embeds = vision_embeds[:video_start_index]
                    video_embeds = vision_embeds[video_start_index:]
                else:
                    raise ValueError(
                        f"Expect video token start index in range [0, {vision_embeds.shape[0]}], but got "
                        f"{video_start_index}"
                    )

                combined_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None,
                )

                if image_embeds is not None or video_embeds is not None:
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                    if image_embeds is not None:
                        image_mask = (input_ids == self.image_token_id).contiguous()
                        if image_mask.sum() > 0:
                            combined_embeddings = combined_embeddings.clone()
                            combined_embeddings[image_mask] = image_embeds.to(
                                dtype=combined_embeddings.dtype, device=combined_embeddings.device
                            )
                    if video_embeds is not None:
                        video_mask = (input_ids == self.video_token_id).contiguous()
                        if video_mask.sum() > 0:
                            combined_embeddings = combined_embeddings.clone()
                            combined_embeddings[video_mask] = video_embeds.to(
                                dtype=combined_embeddings.dtype, device=combined_embeddings.device
                            )
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            else:
                combined_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None,
                )

            if packed_seq_params is not None:
                combined_embeddings = (
                    preprocess_packed_seqs(
                        combined_embeddings.transpose(0, 1).contiguous(), attention_mask, pre_process=True
                    )[0]
                    .transpose(0, 1)
                    .contiguous()
                )
            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None
        from .rope_utils import get_rope_index

        position_ids, _ = get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        if packed_seq_params is not None:
            position_ids = (
                preprocess_packed_seqs(position_ids.permute(1, 2, 0), attention_mask, pre_process=True)[0]
                .permute(2, 0, 1)
                .contiguous()
            )
            attention_mask = None

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
            **kwargs,
        )

        return output
