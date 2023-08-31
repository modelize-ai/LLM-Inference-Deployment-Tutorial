import json
import time
from collections import defaultdict
from enum import Enum
from itertools import chain
from logging import getLogger, Logger
from typing import Dict, List, Optional, Tuple
from threading import Thread

import aiohttp
import requests
from fastapi import HTTPException
from pydantic import BaseModel, Field

from .openai_jumper import *
from protocol.completion_task import *
from protocol.routes import *


class ServerType(Enum):
    SB = "static_batching_server"
    CB = "continuous_batching_server"


class ServerURL:
    def __init__(self, url: str):
        self.url = url
        self.model_id = None
        self.workload = 0
        self.available = True

    @staticmethod
    def _calculate_workload(request_inputs: HuggingFaceCompletionInputs):
        # A simple heuristic method to calculate workload:
        #   - 1.35 here means we assume 1 word ≈ 1.35 tokens
        #   - 20 here means we assume the time consumed to decode 1 token ≈ prefill 20 tokens
        # You can also change the calculation logic here by yourself

        num_prompt_tokens = int(len(request_inputs.prompt.split()) * 1.35)
        num_beams = request_inputs.generation_config.num_beams
        max_new_tokens = request_inputs.generation_config.max_new_tokens

        return num_beams * max_new_tokens + num_prompt_tokens // 20

    def increase_workload(self, request_inputs: HuggingFaceCompletionInputs):
        self.workload += self._calculate_workload(request_inputs)

    def decrease_workload(self, request_inputs: HuggingFaceCompletionInputs):
        self.workload -= self._calculate_workload(request_inputs)

    def __eq__(self, other: "ServerURL"):
        return self.workload == other.workload

    def __gt__(self, other: "ServerURL"):
        return self.workload > other.workload

    def __ge__(self, other: "ServerURL"):
        return self.workload >= other.workload

    def __lt__(self, other: "ServerURL"):
        return self.workload < other.workload

    def __le__(self, other: "ServerURL"):
        return self.workload <= other.workload

    def __hash__(self):
        return hash(self.url)


CLIENT_SINGLETON = None


class ClientNotInitializedError(Exception):
    def __repr__(self):
        return "client is not initialized, please initialize a client object first."

    def __str__(self):
        return self.__repr__()


class ClientDoubleInitializeError(Exception):
    def __repr__(self):
        return "client is initialized, do not initialize again, please use get_client() instead."

    def __str__(self):
        return self.__repr__()


class ClientConfig(BaseModel):
    static_batching_server_urls: Optional[List[str]] = Field(default=None)
    continuous_batching_server_urls: Optional[List[str]] = Field(default=None)
    openai_jumper_configs: Optional[List[OpenAIJumperConfig]] = Field(default=None)
    heart_beat_interval_seconds: int = Field(default=600)


class Client:
    def __init__(self, config: ClientConfig, logger: Optional[Logger] = None):
        global CLIENT_SINGLETON

        if CLIENT_SINGLETON is not None:
            raise ClientDoubleInitializeError()

        self.config = config
        self.logger = logger if logger else getLogger(__name__)

        # containers
        self.model_id2static_batching_server_urls: Dict[str, List[ServerURL]] = defaultdict(list)
        self.model_id2continuous_batching_server_urls: Dict[str, List[ServerURL]] = defaultdict(list)
        self.openai_jumpers: List[OpenAIJumper] = []

        self._update_containers_once()

        Thread(target=self._heart_beat_loop, daemon=True).start()

        # set singleton to self
        CLIENT_SINGLETON = self

    def _update_containers_once(self):
        self._update_server_urls_map(
            static_batching_server_urls=self.config.static_batching_server_urls,
            continuous_batching_server_urls=self.config.continuous_batching_server_urls
        )
        self._update_openai_jumpers(openai_jumper_configs=self.config.openai_jumper_configs)

    def _heart_beat_loop(self):
        while True:
            time.sleep(self.config.heart_beat_interval_seconds)
            self._update_containers_once()

    def _update_server_urls_map(
        self,
        static_batching_server_urls: Optional[List[str]] = None,
        continuous_batching_server_urls: Optional[List[str]] = None
    ):
        def build_server_urls_map(old_server_url_objs: List[ServerURL], new_server_urls: List[str]):
            # TODO: parallelize the execution, the logic here for now may very slow

            server_urls_map = defaultdict(list)
            old_server_url_hash_values = [hash(url_obj) for url_obj in old_server_url_objs]
            for url in new_server_urls:
                hash_value = hash(url)
                if hash_value in old_server_url_hash_values:
                    url_obj = old_server_url_objs[old_server_url_hash_values.index(hash_value)]
                else:
                    url_obj = ServerURL(url=url)
                self.get_model_id(url_obj)
                if url_obj.model_id:
                    server_urls_map[url_obj.model_id].append(url_obj)

            return server_urls_map

        if static_batching_server_urls is not None:
            self.model_id2static_batching_server_urls = build_server_urls_map(
                old_server_url_objs=list(chain(*self.model_id2static_batching_server_urls.values())),
                new_server_urls=static_batching_server_urls
            )

        if continuous_batching_server_urls is not None:
            self.model_id2continuous_batching_server_urls = build_server_urls_map(
                old_server_url_objs=list(chain(*self.model_id2continuous_batching_server_urls)),
                new_server_urls=continuous_batching_server_urls
            )

    def _update_openai_jumpers(self, openai_jumper_configs: Optional[List[OpenAIJumperConfig]] = None):
        if openai_jumper_configs is None:
            return

        old_jumpers = self.openai_jumpers
        old_jumper_hash_values = [hash(jumper) for jumper in old_jumpers]
        new_jumpers = []
        for config in openai_jumper_configs:
            hash_value = hash(config.api_key)
            if hash_value in old_jumper_hash_values:
                jumper = old_jumpers[old_jumper_hash_values.index(hash_value)]
            else:
                jumper = OpenAIJumper(config=config, logger=self.logger)
            new_jumpers.append(jumper)

        self.openai_jumpers = new_jumpers

        for jumper in [jumper for jumper in old_jumpers if jumper not in new_jumpers]:
            destroy_openai_jumper(jumper)

    def update_config(self, config: ClientConfig):
        # One can use this method to implement hot reload logic to update client's behavior on the fly.
        # For example:
        #   1. manually update locally saved config file to update config's parameters;
        #   2. receive an event that a server is startup or shutdown, and automatically update server_urls.

        if set([c.api_key for c in config.openai_jumper_configs]) != \
                set([c.api_key for c in self.config.openai_jumper_configs]):
            self._update_openai_jumpers(openai_jumper_configs=config.openai_jumper_configs)
        new_static_batching_server_urls = None
        new_continuous_batching_server_urls = None
        if set(config.static_batching_server_urls) != set(self.config.static_batching_server_urls):
            new_static_batching_server_urls = config.static_batching_server_urls
        if set(config.continuous_batching_server_urls) != set(self.config.continuous_batching_server_urls):
            new_continuous_batching_server_urls = config.continuous_batching_server_urls
        if new_static_batching_server_urls is not None or new_continuous_batching_server_urls is not None:
            self._update_server_urls_map(
                static_batching_server_urls=new_static_batching_server_urls,
                continuous_batching_server_urls=new_continuous_batching_server_urls
            )
        self.config = config

    def save_config(self, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.config.dict(by_alias=True), f)

    async def openai_chat_completion(
        self,
        request_inputs: OpenAIChatCompletionInputs,
        max_retries: int = 3,
        raise_on_error: bool = True
    ) -> OpenAIChatCompletionOutputs:
        request_inputs.verify_and_preprocess()

        available_jumpers = [jumper for jumper in self.openai_jumpers if jumper.available]
        if not available_jumpers:
            if raise_on_error:
                raise HTTPException(
                    status_code=404,
                    detail="LookupError: none of openai jumper is available for now."
                )
            return OpenAIChatCompletionOutputs()

        jumper = min(available_jumpers)
        request_outputs, error, status_code = await jumper.chat_completion(
            inputs=request_inputs,
            max_retries=max_retries
        )
        if status_code != 200 and raise_on_error:
            raise HTTPException(status_code=status_code, detail=str(error))
        return request_outputs

    async def huggingface_completion(
        self,
        request_inputs: HuggingFaceCompletionInputs,
        max_retries: int = 3,
        raise_on_error: bool = True,
        server_type: ServerType = ServerType.CB,
        timeout: int = 100
    ) -> HuggingFaceCompletionOutputs:
        request_inputs.verify_and_preprocess()

        async def request(
            payload: dict,
            url_obj: ServerURL,
            route: str,
        ) -> Tuple[HuggingFaceCompletionOutputs, Optional[str], int]:
            async with aiohttp.request(
                method="post",
                url=f"{url_obj.url}{route}",
                json=payload,
                headers={},
                timeout=aiohttp.ClientTimeout(timeout)
            ) as resp:
                if resp.status == 200:
                    return HuggingFaceCompletionOutputs(**(await resp.json())), None, resp.status
                else:
                    return HuggingFaceCompletionOutputs(), resp.reason, resp.status

        # check if the requested model_id is available
        model_id = request_inputs.model
        server_urls_map = (
            self.model_id2static_batching_server_urls if server_type == ServerType.SB
            else self.model_id2continuous_batching_server_urls
        )
        url_objs = [url_obj for url_obj in server_urls_map.get(model_id, []) if url_obj.available]
        if not url_objs:
            if raise_on_error:
                raise HTTPException(
                    status_code=404,
                    detail=f"LookupError: requested model [{model_id}] is not available for now."
                )
            return HuggingFaceCompletionOutputs()

        # get the url_obj whose workload is smallest
        url_obj = min(url_objs)

        # request
        url_obj.increase_workload(request_inputs)
        try:
            request_outputs, error, status_code = await request(
                payload=request_inputs.dict(by_alias=True),
                url_obj=url_obj,
                route=ROUTE_POST_STATIC_BATCHING_COMPLETION if server_type == ServerType.SB
                else ROUTE_POST_CONTINUOUS_BATCHING_COMPLETION
            )
        except Exception as e:
            request_outputs = HuggingFaceCompletionOutputs()
            self.logger.error(msg=f"running request method failed with error type [{e.__class__.__name__}]", exc_info=e)
            error = "ServerRequestError: Unknown error occurred when request to server."
            status_code = 500
        url_obj.decrease_workload(request_inputs)

        if status_code == 200:
            return request_outputs
        else:
            if max_retries > 0:
                return await self.huggingface_completion(
                    request_inputs=request_inputs,
                    max_retries=max_retries-1,
                    raise_on_error=raise_on_error,
                    server_type=server_type,
                    timeout=timeout
                )
            elif raise_on_error:
                raise HTTPException(status_code=status_code, detail=error)
            else:
                return request_outputs

    def get_model_id(self, url: ServerURL, max_retries: int = 3) -> Optional[str]:
        res = requests.get(f"{url.url}{ROUTE_GET_MODEL_ID}", timeout=(1, 1), verify=False)
        if res.status_code == 200:
            url.available = True
            url.model_id = res.json()
            return url.model_id
        else:
            if max_retries > 0:
                time.sleep(1)
                return self.get_model_id(url, max_retries - 1)
            self.logger.error(
                msg=f"request to {url.url} to get model_id failed with status [{res.status_code}]"
            )
            url.available = False
            return None


def get_client():
    if CLIENT_SINGLETON is None:
        raise ClientNotInitializedError()
    return CLIENT_SINGLETON


__all__ = [
    "ServerType",
    "Client",
    "ClientConfig",
    "get_client"
]
