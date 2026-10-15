# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# pytorch/torchtune
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in pytorch/torchtune
import logging

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available(check_device=True) -> bool:
    try:
        if not hasattr(torch, "npu"):
            return False

        if check_device:
            return torch.npu.is_available()
        else:
            return True
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def get_visible_devices_keyword() -> str:
    return "CUDA_VISIBLE_DEVICES" if not is_torch_npu_available(check_device=False) else "ASCEND_RT_VISIBLE_DEVICES"


def get_device_name() -> str:
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device():
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    if is_npu_available:
        return "hccl"
    else:
        return "nccl"


def set_expandable_segments(enable: bool) -> None:
    if is_cuda_available:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")


def auto_set_device(config) -> None:
    if config and hasattr(config, "trainer") and hasattr(config.trainer, "device"):
        if is_torch_npu_available():
            if config.trainer.device not in ["cpu", "npu"]:
                logger.warning(
                    f"Detect setting config.trainer.device to {config.trainer.device} for Ascend NPU, maybe"
                    f"from default value in config file, automatically set to `npu` instead."
                )

            config.trainer.device = "npu"


def get_device_capability(device_id: int = 0) -> tuple[int | None, int | None]:
    major, minor = None, None
    if is_cuda_available:
        major, minor = torch.cuda.get_device_capability(device_id)

    return major, minor
