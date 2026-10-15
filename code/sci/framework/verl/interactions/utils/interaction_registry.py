# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import importlib.util
import logging
import os
import sys

from omegaconf import OmegaConf

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_interaction_class(cls_name):
    module_name, class_name = cls_name.rsplit(".", 1)
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    interaction_cls = getattr(module, class_name)
    return interaction_cls


def initialize_interactions_from_config(interaction_config_file):
    interaction_config = OmegaConf.load(interaction_config_file)
    interaction_map = {}

    for interaction_item in interaction_config.interaction:
        cls_name = interaction_item.class_name
        interaction_cls = get_interaction_class(cls_name)

        config = OmegaConf.to_container(interaction_item.config, resolve=True)

        name = interaction_item.get("name", None)
        if name is None:
            class_simple_name = cls_name.split(".")[-1]
            if class_simple_name.endswith("Interaction"):
                name = class_simple_name[:-11].lower()
            else:
                name = class_simple_name.lower()

        if name in interaction_map:
            raise ValueError(f"Duplicate interaction name '{name}' found. Each interaction must have a unique name.")

        config["name"] = name

        interaction = interaction_cls(config=config)
        interaction_map[name] = interaction

        logger.info(f"Initialized interaction '{name}' with class '{cls_name}'")

    return interaction_map
