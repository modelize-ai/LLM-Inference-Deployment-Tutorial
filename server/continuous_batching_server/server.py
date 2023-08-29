import asyncio
import time
import threading
from logging import getLogger, Logger
from typing import *
from uuid import uuid4, UUID

from transformers import AutoTokenizer

from .batcher import Batcher
from .beam import Beam, BeamGroup, BeamStatus
from .worker import Worker
from .config import (
    BatcherConfig,
    CacheConfig,
    ModelLoadingConfig,
    ParallelConfig
)
from protocol.completion_task import (
    TokenUsage,
    HuggingFaceCompletionChoice,
    HuggingFaceGenerationConfig,
    HuggingFaceCompletionInputs,
    HuggingFaceCompletionOutputs
)
from protocol.error import Error


SERVER_SINGLETON = None


class ServerNotInitializedError(Exception):
    def __repr__(self):
        return "server is not initialized, please initialize a server object first."

    def __str__(self):
        return self.__repr__()


class ServerDoubleInitializeError(Exception):
    def __repr__(self):
        return "server is initialized, do not initialize again, please use get_server() instead."

    def __str__(self):
        return self.__repr__()


class Server:
    def __init__(
        self,
        batcher_config: BatcherConfig,
        cache_config: CacheConfig,
        model_loading_config: ModelLoadingConfig,
        parallel_config: ParallelConfig,
        logger: Optional[Logger] = None
    ):
        global SERVER_SINGLETON
        if SERVER_SINGLETON is not None:
            raise ServerDoubleInitializeError()

        assert parallel_config.tp_size == 1, "we don't provide model parallelism support for now."

        self.logger = logger if logger else getLogger(__name__)

        self.worker = Worker(cache_config, model_loading_config, parallel_config, logger)
        self.batcher = Batcher(batcher_config, cache_config, logger)

        self.tokenizer_max_length = model_loading_config.model_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_loading_config.tokenizer_name_or_path or model_loading_config.model_name_or_path,
            use_fast=model_loading_config.use_fast_tokenizer,
            trust_remote_code=model_loading_config.trust_remote_code,
            truncation_side="left",
        )
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.finished_table: Dict[UUID, Tuple[HuggingFaceCompletionOutputs, Optional[Error], int]] = dict()

        threading.Thread(target=self._run, daemon=True).start()

        SERVER_SINGLETON = self

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer(text, truncation=True, max_length=self.tokenizer_max_length)["input_ids"]

    def _decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _construct_generation_result(
            self,
            request: BeamGroup
    ) -> Tuple[HuggingFaceCompletionOutputs, Optional[Error], int]:
        final_beams = request.get_final_beams()
        choices = [
            HuggingFaceCompletionChoice(
                text=self._decode(beam.generated_token_ids),
                index=idx,
                finish_reason=beam.finish_reason
            )
            for idx, beam in enumerate(final_beams)
        ]
        usage = TokenUsage(
            prompt_tokens=len(request.get_final_beams()[0].prompt_token_ids),
            completion_tokens=sum([beam.num_generated_tokens for beam in request.get_beams(BeamStatus.FINISHED)])
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return HuggingFaceCompletionOutputs(choices=choices, usage=usage), None, 200

    async def wait_task_done(
            self,
            inp: HuggingFaceCompletionInputs
    ) -> Tuple[HuggingFaceCompletionOutputs, Optional[Error], int, float]:
        request_id = uuid4()
        start = time.time()

        inp.generation_config.eos_token_id = self.tokenizer.eos_token_id
        inp.generation_config.pad_token_id = self.tokenizer.pad_token_id

        request = BeamGroup(
            request_id=request_id,
            arrival_time=time.time(),
            beams=[
                Beam(
                    request_id,
                    prompt=inp.prompt,
                    prompt_token_ids=self._encode(inp.prompt),
                    block_size=self.batcher.cache_config.block_size
                )
            ],
            generation_config=inp.generation_config
        )
        self.logger.info(msg=f"Task-{request_id} is added.")
        self.batcher.add_request(request)

        while True:
            await asyncio.sleep(0.1)
            if request_id in self.finished_table:
                end = time.time()
                outputs, error, status_code = self.finished_table.pop(request_id)
                wall_time = end - start
                self.logger.info(msg=f"Task-{request_id} is finished, {wall_time=: .4f}s")
                return outputs, error, status_code, wall_time

    def _run(self) -> None:
        steps = 0
        while True:
            steps += 1
            batch, blocks_to_copy, blocks_to_swap_in, blocks_to_swap_out, finished_requests = self.batcher.schedule()
            for req in finished_requests:
                self.finished_table[req.request_id] = self._construct_generation_result(req)
            self.worker.forward(batch, blocks_to_copy, blocks_to_swap_in, blocks_to_swap_out)
            self.batcher.batch_generate(batch)
            time.sleep(0.001)


def get_server():
    if SERVER_SINGLETON is None:
        raise ServerNotInitializedError()
    return SERVER_SINGLETON


__all__ = ["Server", "get_server"]
