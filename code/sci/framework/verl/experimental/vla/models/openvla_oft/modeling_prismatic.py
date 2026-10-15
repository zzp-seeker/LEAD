# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from PRIME-RL/SimpleVLA-RL
# form https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero10-trajall/blob/main/
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Optional

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig
from .constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
)
from .train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)

logger = logging.getLogger(__name__)


def unpack_tuple(fn: Callable[[Any], tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


class PrismaticVisionBackbone(nn.Module):

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: list[int],
        timm_model_ids: list[str],
        timm_override_act_layers: list[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1

        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic models only support up to 2 (fused) vision backbones!")

        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim

        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        self._patch_layer_scales()

    def _create_featurizer(self, model_id: str, img_size: int, act_layer: Optional[str]) -> nn.Module:
        featurizer = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

        return featurizer

    def _patch_layer_scales(self) -> None:
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def get_num_patches(self) -> int:
        return self.featurizer.patch_embed.num_patches

    def get_num_images_in_input(self) -> int:
        return self.num_images_in_input

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        self.num_images_in_input = num_images_in_input

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.num_images_in_input == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

            return torch.cat([patches, patches_fused], dim=2)

        else:
            assert self.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            images = torch.split(pixel_values, [6] * self.num_images_in_input, dim=1)

            all_patches = []
            for img in images:
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                patches = self.featurizer(img_regular)
                patches_fused = self.fused_featurizer(img_fused)

                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            return torch.cat(all_patches, dim=1)


class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None

    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[list[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear | nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please report the issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.llm_dim = config.text_config.hidden_size

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def _replace_input_embeddings(self, input_embeddings, all_actions_mask, noisy_action_features):
        new_input_embeddings = input_embeddings.clone()

        repositioned_noisy_action_features = torch.zeros_like(input_embeddings)

        batch_indices = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, noisy_action_features.shape[1])

        masked_indices = torch.stack([torch.where(mask)[0] for mask in all_actions_mask])

        repositioned_noisy_action_features[batch_indices, masked_indices] = noisy_action_features

        new_input_embeddings = torch.where(
            all_actions_mask.unsqueeze(-1), repositioned_noisy_action_features, new_input_embeddings
        )

        return new_input_embeddings

    def _process_action_masks(self, labels):
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask
        return all_actions_mask

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        if use_film:
            patch_features = self.vision_backbone(pixel_values, language_embeddings)
        else:
            patch_features = self.vision_backbone(pixel_values)

        return self.projector(patch_features)

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        if proprio_projector is not None and proprio is not None:
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)
            proprio_features = proprio_projector(proprio)
            proprio_features = proprio_features.unsqueeze(dim=1)
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )

        return multimodal_embeddings, multimodal_attention_mask

    def _build_multimodal_labels(self, labels, projected_patch_embeddings):
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        return None




























    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> dict[str, torch.Tensor]:
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)

    def _prepare_input_for_action_prediction_verl(self, input_ids, attention_mask):
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction_verl(self, labels, input_ids):
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        labels[:, -1] = STOP_INDEX

        return labels

    def _verl_discrete_compute_logits(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
    ):
        all_actions_mask = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask

        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )



        compute_logits = language_model_output.logits[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
        ]

        return compute_logits

















































    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values=None,
        attention_mask=None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ):


        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1


        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        labels[:, -1] = STOP_INDEX



        input_embeddings = self.get_input_embeddings()(input_ids)

        newline_positions = labels != IGNORE_INDEX

        cumsum = torch.cumsum(newline_positions, dim=1)

        mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

        action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
        current_action_mask = action_tokens_only_mask * mask

        newline_positions = labels != IGNORE_INDEX

        cumsum = torch.cumsum(newline_positions, dim=1)

        mask = cumsum > ACTION_DIM

        action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
        next_actions_mask = action_tokens_only_mask * mask

        all_actions_mask = current_action_mask | next_actions_mask


        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        if use_film:
            raise ValueError
            patch_features = self.vision_backbone(pixel_values, language_embeddings)
        else:
            patch_features = self.vision_backbone(pixel_values)

        projected_patch_embeddings = self.projector(patch_features)

        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(
                projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype
            )
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            raise ValueError
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
            )
        else:


            all_actions_mask = all_actions_mask.unsqueeze(-1)
            input_embeddings = input_embeddings * ~all_actions_mask


            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )

            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )



            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            compute_logits = language_model_output.logits[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            ]


        return compute_logits


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig
    _supports_sdpa = True

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions

    def _run_diffusion_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        noise,
        action_head,
        projected_patch_embeddings,
        labels,
        attention_mask,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        noisy_action_projector,
    ):
        action_head.noise_scheduler.set_timesteps(action_head.num_diffusion_steps)
        orig_projected_patch_embeddings = projected_patch_embeddings.clone()
        curr_noisy_actions = noise

        for t in action_head.noise_scheduler.timesteps:
            timesteps = torch.Tensor([t]).to(labels.device)
            diffusion_timestep_embeddings = (
                action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)


            projected_patch_embeddings = torch.cat(
                (orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
            )

            B = curr_noisy_actions.shape[0]
            orig_curr_noisy_actions_shape = curr_noisy_actions.shape
            curr_noisy_actions = curr_noisy_actions.reshape(B, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(curr_noisy_actions)
            curr_noisy_actions = curr_noisy_actions.reshape(orig_curr_noisy_actions_shape)

            input_embeddings = self._replace_input_embeddings(
                input_embeddings.clone(), all_actions_mask, noisy_action_features
            )

            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            last_hidden_states = language_model_output.hidden_states[-1]
            actions_hidden_states = last_hidden_states[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                :,
            ]

            noise_pred = action_head.predict_noise(actions_hidden_states)
            curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states

    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
    ):
        all_actions_mask = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask

        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_states = language_model_output.hidden_states[-1]
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]

        if action_head is not None:
            normalized_actions = action_head.predict_action(actions_hidden_states)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states

    def _verl_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        do_sample=True,
        temperature=1,
    ):
        all_actions_mask = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask

        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )




        NUM_PROMPT_TOKENS = NUM_PROMPT_TOKENS + NUM_PATCHES
        batch_size = language_model_output.logits.shape[0]
        device = language_model_output.logits.device

        start_indices = NUM_PROMPT_TOKENS.unsqueeze(1)
        position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=device).unsqueeze(0)
        seq_indices = start_indices + position_offsets
        if not do_sample:


            reponse_ids_logits = language_model_output.logits[
                torch.arange(batch_size, device=device).unsqueeze(-1), seq_indices, :
            ]
            start_index = self.vocab_size - 256
            response_last256 = reponse_ids_logits[..., -256 - 64 : -64]
            last256_argmax = response_last256.argmax(dim=-1)
            reponse_ids = last256_argmax + start_index

            predicted_action_token_ids = reponse_ids.cpu().numpy()

        else:
            assert temperature > 0

            action_logits = language_model_output.logits[
                torch.arange(batch_size, device=device).unsqueeze(-1), seq_indices, :
            ]

            action_logits_last256 = action_logits[..., -256 - 64 : -64]
            scaled_logits = action_logits_last256 / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            assert probs.shape[-1] == 256
            probs_flat = probs.reshape(-1, probs.shape[-1])
            sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
            original_ids_flat = sampled_indices_flat + (self.vocab_size - 256)
            reponse_ids = original_ids_flat.view(action_logits.shape[0], -1)

            predicted_action_token_ids = reponse_ids.cpu().numpy()

        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        normalized_actions = normalized_actions.reshape(-1, ACTION_DIM)

        return normalized_actions, reponse_ids

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]

        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1

        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(
                projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype
            )
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
            )
        else:
            normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                action_head,
            )

        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return actions, actions_hidden_states

    def generate_action_verl(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) -> np.ndarray:

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]
        do_sample = kwargs["do_sample"]
        temperature = kwargs["temperature"]

        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        padding_idx = kwargs["padding_idx"]
        num_prompt_tokens = input_ids.ne(padding_idx).sum(dim=1) - 1

        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        padding_mask = input_ids.ne(padding_idx)
        assert torch.all(padding_mask == attention_mask.ne(0))
        padding_mask = padding_mask.int()
        sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
        input_ids = torch.gather(input_ids, 1, sorted_indices)
        attention_mask = torch.gather(attention_mask, 1, sorted_indices)
        labels = torch.gather(labels, 1, sorted_indices)
        assert not use_film

        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(
                projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype
            )
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            raise ValueError

        else:
            normalized_actions, reponse_ids = self._verl_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                num_prompt_tokens,
                action_head,
                do_sample=do_sample,
                temperature=temperature,
            )

        actions = self._unnormalize_actions(normalized_actions, unnorm_key)
        actions = actions.reshape(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)
        return actions, reponse_ids

    @staticmethod
    def _check_unnorm_key(norm_stats: dict[str, dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> dict[str, Any]:
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
