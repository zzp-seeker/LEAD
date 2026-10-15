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
import json
from typing import Any

VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"


def get_vllm_max_lora_rank(lora_rank: int):
    assert lora_rank > 0, f"lora_rank must be greater than 0 to invoke this function, get {lora_rank}"
    vllm_max_lora_ranks = [8, 16, 32, 64, 128, 256, 320, 512]
    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank

    raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                continue
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args
