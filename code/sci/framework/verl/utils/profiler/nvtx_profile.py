# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
import functools
from contextlib import contextmanager
from typing import Callable, Optional

import nvtx
import torch

from .config import NsightToolConfig
from .profile import DistProfiler, ProfilerConfig


def mark_start_range(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    return nvtx.start_range(message=message, color=color, domain=domain, category=category)


def mark_end_range(range_id: str) -> None:
    return nvtx.end_range(range_id)


def mark_annotate(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> Callable:

    def decorator(func):
        profile_message = message or func.__name__
        return nvtx.annotate(profile_message, color=color, domain=domain, category=category)(func)

    return decorator


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    mark_range = mark_start_range(message=name, color=color, domain=domain, category=category)
    from .performance import _timer

    yield from _timer(name, timing_raw)
    mark_end_range(mark_range)


class NsightSystemsProfiler(DistProfiler):

    def __init__(self, rank: int, config: Optional[ProfilerConfig], tool_config: Optional[NsightToolConfig], **kwargs):
        if not config:
            config = ProfilerConfig(ranks=[])
        if not tool_config:
            assert not config.enable, "tool_config must be provided when profiler is enabled"
        self.enable = config.enable
        if not config.enable:
            return
        self.this_step: bool = False
        self.discrete: bool = tool_config.discrete
        self.this_rank: bool = False
        if config.all_ranks:
            self.this_rank = True
        elif config.ranks:
            self.this_rank = rank in config.ranks

    def start(self, **kwargs):
        if self.enable and self.this_rank:
            self.this_step = True
            if not self.discrete:
                torch.cuda.profiler.start()

    def stop(self):
        if self.enable and self.this_rank:
            self.this_step = False
            if not self.discrete:
                torch.cuda.profiler.stop()

    def annotate(
        self,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs_outer,
    ) -> Callable:

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                if not self.enable:
                    return func(*args, **kwargs_inner)

                profile_name = message or func.__name__

                if self.this_step:
                    if self.discrete:
                        torch.cuda.profiler.start()
                    mark_range = mark_start_range(message=profile_name, color=color, domain=domain, category=category)

                result = func(*args, **kwargs_inner)

                if self.this_step:
                    mark_end_range(mark_range)
                    if self.discrete:
                        torch.cuda.profiler.stop()

                return result

            return wrapper

        return decorator
