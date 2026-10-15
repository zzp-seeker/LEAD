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
import asyncio
import logging
from collections import deque
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        self.queue = deque(maxlen=self.max_queue_size)
        self.current_param_version = 0

        self.val_queue = deque()

        try:
            if hasattr(config, "async_training") and config.async_training is not None:
                self.staleness_threshold = getattr(config.async_training, "staleness_threshold", 3)
            else:
                self.staleness_threshold = 3
        except (AttributeError, RecursionError):
            self.staleness_threshold = 3

        self.running = True

        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        print(
            f"[MessageQueue] initialized with max_queue_size={max_queue_size},"
            f"staleness_threshold={self.staleness_threshold}"
        )

    async def put_sample(self, sample: Any, param_version: int) -> bool:
        async with self._lock:
            is_drop = False
            if len(self.queue) >= self.max_queue_size:
                self.queue.popleft()
                self.dropped_samples += 1
                is_drop = True
                logger.warning("Queue full, dropped sample")
            self.queue.append(sample)
            self.total_produced += 1

            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            if is_drop:
                return False
            return True

    async def get_sample(self) -> Any | None:
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            if not self.running and len(self.queue) == 0:
                return None

            data = self.queue.popleft()
            self.total_consumed += 1
            return data, len(self.queue)

    async def update_param_version(self, version: int):
        async with self._lock:
            old_version = self.current_param_version
            self.current_param_version = version
            print(f"Parameter version updated from {old_version} to {version}")

    async def get_queue_size(self) -> int:
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "current_param_version": self.current_param_version,
                "staleness_threshold": self.staleness_threshold,
                "max_queue_size": self.max_queue_size,
            }

    async def clear_queue(self):
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        async with self._lock:
            self.running = False
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        async with self._lock:
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                sample = list(self.queue)[0]
                try:
                    sample_size = sys.getsizeof(sample)
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000
                    if hasattr(sample, "agent_loop_output"):
                        sample_size += 5000
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }

    async def put_validate(self, data):
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any, param_version: int) -> bool:
        future = self.queue_actor.put_sample.remote(sample, param_version)
        return await asyncio.wrap_future(future.future())

    async def put_validate(self, data: Any) -> bool:
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_sample(self) -> Any | None:
        future = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    def put_sample_sync(self, sample: Any, param_version: int) -> bool:
        return ray.get(self.queue_actor.put_sample.remote(sample, param_version))

    def get_sample_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        return ray.get(self.queue_actor.get_statistics.remote())

    def update_param_version_sync(self, version: int):
        return ray.get(self.queue_actor.update_param_version.remote(version))
