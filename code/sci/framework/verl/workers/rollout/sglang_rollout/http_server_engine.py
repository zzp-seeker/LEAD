# Copyright 2025 z.ai
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
#
# This file is adapted from multiple sources:
# 1. THUDM/slime project
#    Original source: THUDM/slime
#    Copyright 2025 z.ai
#    Licensed under the Apache License, Version 2.0
# 2. SGLang project
#    Original source: sgl-project/sglang
#    Copyright 2023-2024 SGLang Team
#    Licensed under the Apache License, Version 2.0
#
# Modifications made by z.ai and ModelBest Inc. include but are not limited to:
# - Enhanced error handling and retry logic
# - Added async support with connection pooling
# - Extended functionality for distributed weight updates
# - Improved logging and monitoring capabilities
# - Additional configuration options and optimizations
import asyncio
import logging
import multiprocessing
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

import aiohttp
import requests
from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.managers.io_struct import (
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_MAX_CONNECTIONS = 2000
DEFAULT_MAX_WAIT_TIME = 300.0


def _read_response(response: requests.Response):
    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError:
        return {
            "content_type": response.headers.get("Content-Type", ""),
            "text": response.text,
        }


async def _read_async_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
    if resp.status == 204 or (resp.content_length == 0):
        return {}

    try:
        return await resp.json(content_type=None)
    except Exception:
        try:
            text = await resp.text()
        except Exception:
            return {}
        return {
            "content_type": (resp.headers.get("Content-Type") or ""),
            "text": text,
        }


def launch_server_process(
    server_args: ServerArgs,
    timeout: float = DEFAULT_TIMEOUT,
    max_wait_time=DEFAULT_MAX_WAIT_TIME,
    first_rank_in_node=False,
) -> multiprocessing.Process:
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    if server_args.node_rank != 0 or not first_rank_in_node:
        logger.info(f"Server process started with PID {p.pid} for node rank {server_args.node_rank}", flush=True)
        return p

    p.start()

    base_url = server_args.url()
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {server_args.api_key}",
    }

    start_time = time.time()

    with requests.Session() as session:
        while time.time() - start_time < max_wait_time:
            if not p.is_alive():
                raise RuntimeError("Server process terminated unexpectedly during startup")

            try:
                if server_args.is_embedding:
                    response = session.get(f"{base_url}/health", headers=headers, timeout=timeout)
                else:
                    response = session.get(f"{base_url}/health_generate", headers=headers, timeout=timeout)
                if response.status_code == 200:
                    break
            except requests.RequestException as e:
                logger.debug(f"Health check failed: {e}")

            time.sleep(2)
        else:
            p.terminate()
            logger.error(f"Server in {base_url} failed to become healthy within timeout period")
            raise TimeoutError("Server failed to become healthy within timeout period")

        while time.time() - start_time < max_wait_time:
            if not p.is_alive():
                raise RuntimeError("Server process terminated unexpectedly during cache flush")

            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers, timeout=timeout)
                if response.status_code == 200:
                    break
            except requests.RequestException as e:
                logger.debug(f"Cache flush check failed: {e}")

            time.sleep(2)
        else:
            p.terminate()
            raise TimeoutError("Server cache flush failed within timeout period")

    return p


class HttpServerAdapter(EngineBase):

    def __init__(
        self,
        router_ip: Optional[str] = None,
        router_port: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        first_rank_in_node: bool = False,
        max_start_wait_time: float = DEFAULT_MAX_WAIT_TIME,
        launch_server: bool = True,
        **kwargs: Any,
    ) -> None:
        self.router_ip: Optional[str] = router_ip
        self.router_port: Optional[int] = router_port
        self.timeout: float = timeout
        self.max_attempts: int = max_attempts
        self.retry_delay: float = retry_delay
        self.server_args: ServerArgs = ServerArgs(**kwargs)
        self.node_rank: int = self.server_args.node_rank
        self.max_start_wait_time: float = max_start_wait_time

        logger.info(
            f"Launch HttpServerAdapter at: {self.server_args.host}:{self.server_args.port} with {first_rank_in_node}"
        )
        if launch_server:
            self.process: multiprocessing.Process = launch_server_process(
                self.server_args, self.timeout, self.max_start_wait_time, first_rank_in_node
            )

        if self.node_rank == 0 and self.router_ip and self.router_port:
            self._register_with_router()

    def _register_with_router(self) -> None:
        try:
            url = f"http://{self.router_ip}:{self.router_port}/add_worker"
            params = {"url": f"http://{self.server_args.host}:{self.server_args.port}"}
            response = requests.post(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            logger.info("Successfully registered with router")
        except Exception as e:
            logger.error(f"Failed to register with router: {e}")

    def _make_request(
        self,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
        method: str = "POST",
        timeout: float = DEFAULT_TIMEOUT,
        only_master: bool = True,
    ) -> dict[str, Any]:
        if only_master and self.node_rank != 0:
            return {}

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"

        for attempt in range(self.max_attempts):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, timeout=self.timeout)
                else:
                    response = requests.post(url, json=payload or {}, timeout=self.timeout)

                response.raise_for_status()
                return _read_response(response)

            except requests.exceptions.Timeout:
                logger.warning(f"Request to {endpoint} timed out (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error for {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {e}")
                if attempt == self.max_attempts - 1:
                    raise

            if attempt < self.max_attempts - 1:
                time.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(f"Failed to complete request to {endpoint} after {self.max_attempts} attempts")

    def update_weights_from_tensor(self, req: UpdateWeightsFromTensorReqInput) -> dict[str, Any]:
        import base64

        named_tensors = req.serialized_named_tensors
        load_format = req.load_format
        flush_cache = req.flush_cache

        if named_tensors:
            serialized_named_tensors = [
                base64.b64encode(named_tensor).decode("utf-8") for named_tensor in named_tensors
            ]
        else:
            serialized_named_tensors = []

        return self._make_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    def shutdown(self) -> None:
        if self.router_ip and self.router_port:
            try:
                url = f"http://{self.router_ip}:{self.router_port}/remove_worker"
                params = {"url": f"http://{self.server_args.host}:{self.server_args.port}"}
                requests.post(url, params=params, timeout=5.0)
                logger.info("Successfully unregistered from router")
            except Exception as e:
                logger.warning(f"Failed to unregister from router: {e}")

        if hasattr(self, "process") and self.process is not None:
            try:
                kill_process_tree(self.process.pid)
                logger.info("Server process terminated")
            except Exception as e:
                logger.error(f"Failed to terminate server process: {e}")

    def generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[dict[str, Any]] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        token_ids_logprob: Optional[list[int]] = None,
        lora_path: Optional[str] = None,
        custom_logit_processor: Optional[Callable] = None,
    ) -> dict[str, Any]:
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        return self._make_request("generate", payload, only_master=False)

    def reward_score(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        lora_path: Optional[str] = None,
    ) -> dict[str, Any]:
        assert self.server_args.is_embedding, "Score is only supported for embedding models"
        payload = {
            "text": prompt,
            "input_ids": input_ids,
            "image_data": image_data,
            "lora_path": lora_path,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        return self._make_request("classify", payload, only_master=False)

    def flush_cache(self) -> dict[str, Any]:
        if self.node_rank != 0:
            return {}

        for attempt in range(self.max_attempts * 2):
            try:
                response = requests.get(
                    f"http://{self.server_args.host}:{self.server_args.port}/flush_cache", timeout=self.timeout
                )
                if response.status_code == 200:
                    return _read_response(response)
            except Exception as e:
                logger.warning(f"Error flushing cache (attempt {attempt + 1}): {e}")

            time.sleep(self.retry_delay)

        logger.error("Failed to flush cache after maximum attempts")
        return {}

    def release_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        return self._make_request("release_memory_occupation", {"tags": tags})

    def resume_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        return self._make_request("resume_memory_occupation", {"tags": tags})

    def abort_request(self, rid: str = "", abort_all: bool = False) -> dict[str, Any]:
        return self._make_request("abort_request", {"rid": rid, "abort_all": abort_all})


class AsyncHttpServerAdapter(HttpServerAdapter):

    def __init__(
        self,
        router_ip: Optional[str] = None,
        router_port: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        first_rank_in_node: bool = False,
        launch_server: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            router_ip,
            router_port,
            timeout,
            max_attempts,
            retry_delay,
            first_rank_in_node,
            launch_server=launch_server,
            **kwargs,
        )
        self.max_connections: int = max_connections

    @asynccontextmanager
    async def _get_session(self) -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections // 4,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        try:
            yield session
        finally:
            if not session.closed:
                await session.close()

    async def _make_async_request(
        self,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
        method: str = "POST",
        timeout: float = DEFAULT_TIMEOUT,
        only_master: bool = True,
    ) -> dict[str, Any]:
        if only_master and self.node_rank != 0:
            return {}

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"

        for attempt in range(self.max_attempts):
            try:
                async with self._get_session() as session:
                    if method.upper() == "GET":
                        async with session.get(url, timeout=timeout) as response:
                            response.raise_for_status()
                            return await _read_async_response(response)
                    else:
                        async with session.post(url, json=payload or {}, timeout=timeout) as response:
                            response.raise_for_status()
                            return await _read_async_response(response)

            except asyncio.TimeoutError:
                logger.warning(f"Async request to {endpoint} timed out (attempt {attempt + 1})")
            except aiohttp.ClientConnectorError:
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP error for {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {e}")
                if attempt == self.max_attempts - 1:
                    raise

            if attempt < self.max_attempts - 1:
                await asyncio.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(f"Failed to complete async request to {endpoint} after {self.max_attempts} attempts")

    async def release_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        return await self._make_async_request("release_memory_occupation", {"tags": tags})

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        return await self._make_async_request("resume_memory_occupation", {"tags": tags})

    async def update_weights_from_tensor(
        self,
        req: UpdateWeightsFromTensorReqInput,
    ) -> dict[str, Any]:
        import base64

        named_tensors = req.serialized_named_tensors
        load_format = req.load_format
        flush_cache = req.flush_cache

        serialized_named_tensors = [base64.b64encode(named_tensor).decode("utf-8") for named_tensor in named_tensors]
        return await self._make_async_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    async def flush_cache(self) -> dict[str, Any]:
        if self.node_rank != 0:
            return {}

        for attempt in range(self.max_attempts * 4):
            try:
                async with self._get_session() as session:
                    url = f"http://{self.server_args.host}:{self.server_args.port}/flush_cache"
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await _read_async_response(response)
            except Exception as e:
                logger.warning(f"Error flushing cache (attempt {attempt + 1}): {e}")

            await asyncio.sleep(self.retry_delay)

        logger.error("Failed to flush cache after maximum attempts")
        return {}

    async def generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[dict[str, Any]] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        token_ids_logprob: Optional[list[int]] = None,
        lora_path: Optional[str] = None,
        custom_logit_processor: Optional[Callable] = None,
    ) -> dict[str, Any]:
        logger.info("generate() started")

        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        response = await self._make_async_request("generate", payload, timeout=self.timeout, only_master=False)

        return response

    async def async_generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[dict[str, Any]] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        token_ids_logprob: Optional[list[int]] = None,
        lora_path: Optional[str] = None,
        custom_logit_processor: Optional[Callable] = None,
    ) -> dict[str, Any]:
        return await self.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            input_ids=input_ids,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
        )

    async def reward_score(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        lora_path: Optional[str] = None,
    ) -> dict[str, Any]:
        logger.info("reward_score() started")
        payload = {
            "text": prompt,
            "input_ids": input_ids,
            "image_data": image_data,
            "lora_path": lora_path,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        response = await self._make_async_request("classify", payload, timeout=self.timeout, only_master=False)

        return response

    async def async_reward_score(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[list[int]] = None,
        image_data: Optional[Any] = None,
        lora_path: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self.reward_score(
            prompt=prompt,
            input_ids=input_ids,
            image_data=image_data,
            lora_path=lora_path,
        )

    async def abort_request(self, rid: str = "", abort_all: bool = False) -> dict[str, Any]:
        return await self._make_async_request("abort_request", {"rid": rid, "abort_all": abort_all})
