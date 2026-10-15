# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import warnings
from enum import Enum

import torch

try:
    from megatron.core.transformer.moe.moe_utils import (
        apply_router_token_dropping,
        compute_routing_scores_for_aux_loss,
        group_limited_topk,
    )
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
    pass
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig



class RouterReplayAction(Enum):
    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class RouterReplay:

    router_instances = []

    @staticmethod
    def set_replay_data(all_layers_topk_indices: list):
        if len(all_layers_topk_indices) != len(RouterReplay.router_instances):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match the number of router instances ({len(RouterReplay.router_instances)})."
            )
        for i, router_instance in enumerate(RouterReplay.router_instances):
            router_instance.set_target_indices(all_layers_topk_indices[i])

    @staticmethod
    def get_recorded_data() -> list:
        return [router.get_recorded_indices() for router in RouterReplay.router_instances]

    @staticmethod
    def clear_global_indices():
        for router in RouterReplay.router_instances:
            router.clear_indices()

    def __init__(self):
        self.target_topk_idx = None
        self.recorded_topk_idx = None
        self.router_replay_action = None
        self.replay_backward_list = []
        RouterReplay.router_instances.append(self)

    def set_target_indices(self, topk_indices: torch.Tensor):
        self.target_topk_idx = topk_indices
        self.replay_backward_list.append(topk_indices)

    def get_recorded_indices(self):
        return self.recorded_topk_idx

    def record_indices(self, topk_indices: torch.Tensor):
        self.recorded_topk_idx = topk_indices

    def clear_indices(self):
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_backward_list = []

    def set_router_replay_action(self, router_replay_action: RouterReplayAction):
        self.router_replay_action = router_replay_action

    def clear_router_replay_action(self):
        self.router_replay_action = None

    @staticmethod
    def set_global_router_replay_action(router_replay_action: RouterReplayAction):
        for router in RouterReplay.router_instances:
            router.set_router_replay_action(router_replay_action)

    @staticmethod
    def clear_global_router_replay_action():
        for router in RouterReplay.router_instances:
            router.clear_router_replay_action()


def _patched_topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    score_function: str,
    expert_bias: torch.Tensor,
    fused: bool,
    router_replay: RouterReplay,
    scaling_factor: float,
):
    num_tokens, num_experts = logits.shape

    def _compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    def compute_topk(scores, topk, num_groups=None, group_topk=None):

        routing_action = router_replay.router_replay_action if router_replay is not None else None

        if routing_action is None:
            return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

        if routing_action == RouterReplayAction.RECORD:
            probs, top_indices = _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if router_replay is not None:
                router_replay.record_indices(top_indices)
            return probs, top_indices

        elif routing_action == RouterReplayAction.REPLAY_FORWARD:
            if router_replay is None or router_replay.target_topk_idx is None:
                return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

            top_indices = router_replay.target_topk_idx
            top_indices = top_indices.to(scores.device)
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        elif routing_action == RouterReplayAction.REPLAY_BACKWARD:
            if router_replay is None or not router_replay.replay_backward_list:
                return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

            top_indices = router_replay.replay_backward_list.pop(0)
            top_indices = top_indices.to(scores.device)
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    if torch.are_deterministic_algorithms_enabled():
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)

        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_((rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False)
        routing_map = routing_map.bool()
    else:
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return routing_probs, routing_map


def patched_routing(self, logits: torch.Tensor):
    seq_length, bsz = logits.shape[:2]
    logits = logits.view(-1, self.config.num_moe_experts)

    logits = self.apply_z_loss(logits)

    if self.routing_type == "sinkhorn":
        probs, routing_map = self.sinkhorn_load_balancing(logits)
    else:
        probs, routing_map = _patched_topk_routing_with_score_function(
            logits=logits,
            topk=self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            fused=self.config.moe_router_fusion,
            router_replay=self.router_replay,
        )

    if self.config.moe_expert_capacity_factor is not None:
        probs, routing_map = apply_router_token_dropping(
            probs,
            routing_map,
            router_topk=self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            drop_policy=self.config.moe_token_drop_policy,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        )

    if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
        routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
            logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
        )
        probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
        probs = self._apply_seq_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz)
        probs = self._apply_global_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)

    if self.enable_expert_bias and torch.is_grad_enabled():
        with torch.no_grad():
            self.local_tokens_per_expert += routing_map.sum(dim=0)

    return probs, routing_map


def apply_router_replay_patch():
    print("Applying Router Replay Patch...")
    RouterReplay.router_instances.clear()
    if not hasattr(TransformerConfig, "enable_routing_replay"):
        TransformerConfig.enable_routing_replay = False

        original_tf_config_init = TransformerConfig.__init__

        def patched_tf_config_init(self, *args, **kwargs):
            enable_routing_replay = kwargs.pop("enable_routing_replay", TransformerConfig.enable_routing_replay)

            original_tf_config_init(self, *args, **kwargs)

            self.enable_routing_replay = enable_routing_replay

        TransformerConfig.__init__ = patched_tf_config_init

    if hasattr(TopKRouter, "_router_replay_patched"):
        return

    original_init = TopKRouter.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.router_replay = None
        if self.config.enable_routing_replay:
            self.router_replay = RouterReplay()

    TopKRouter.__init__ = patched_init
    TopKRouter.routing = patched_routing
    TopKRouter._router_replay_patched = True
