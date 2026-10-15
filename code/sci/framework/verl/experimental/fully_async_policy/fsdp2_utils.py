# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import torch.distributed as dist
from packaging import version
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec

if version.parse(torch.__version__) < version.parse("2.6"):
    raise RuntimeError("PyTorch 2.6 or higher is required to use fstp_utils.")


def fsdp2_sharded_save_to_cpu(
    model: torch.nn.Module,
) -> tuple[dict[str, tuple[torch.Tensor, DTensorSpec]], DTensorSpec]:
    cpu_sharded_state = {}
    global_spec = None

    for param_name, param in model.named_parameters():
        if not isinstance(param, DTensor):
            cpu_tensor = param.detach().cpu()
            cpu_sharded_state[param_name] = (cpu_tensor, None)
            continue

        if global_spec is None:
            global_spec = param._spec
            assert hasattr(global_spec, "device_mesh"), "DTensorSpec must contain 'device_mesh' attribute"
            assert hasattr(global_spec, "placements"), "DTensorSpec must contain 'placements' attribute"

        local_gpu_tensor = param._local_tensor
        local_cpu_tensor = local_gpu_tensor.detach().cpu()
        cpu_sharded_state[param_name] = (local_cpu_tensor, param._spec)

    assert global_spec is not None, "No DTensor-type parameters found in the model. FSDP2 sharding may not be enabled."
    return cpu_sharded_state, global_spec


def fsdp2_sharded_load_from_cpu(
    model: torch.nn.Module,
    cpu_sharded_state: dict[str, tuple[torch.Tensor, Optional[DTensorSpec]]],
    target_spec: DTensorSpec,
) -> None:
    current_device_mesh = None
    for param in model.parameters():
        if isinstance(param, DTensor):
            current_device_mesh = param._spec.device_mesh
            break
    assert current_device_mesh is not None, "DTensor parameters not initialized in the model to be loaded"
    assert current_device_mesh == target_spec.device_mesh, (
        f"device_mesh mismatch during loading! Original: {target_spec.device_mesh}, Current: {current_device_mesh}"
    )

    for param_name, param in model.named_parameters():
        if param_name not in cpu_sharded_state:
            continue

        local_cpu_tensor, saved_spec = cpu_sharded_state[param_name]

        if isinstance(param, DTensor):
            assert saved_spec is not None, f"DTensorSpec missing in saved state for parameter {param_name}"
            assert saved_spec.placements == target_spec.placements, (
                f"Sharding strategy mismatch for parameter {param_name} (conflicts with global rules)!"
            )

            target_device = param._local_tensor.device
            local_gpu_tensor = local_cpu_tensor.to(target_device)

            param._local_tensor.copy_(local_gpu_tensor)

        else:
            target_device = param.device
            param.data.copy_(local_cpu_tensor.to(target_device))

    dist.barrier()
