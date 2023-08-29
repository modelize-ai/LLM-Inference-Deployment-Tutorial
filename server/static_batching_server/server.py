import asyncio
import time
from logging import getLogger, Logger
from typing import Dict, Optional, Tuple
from threading import Thread
from uuid import uuid4, UUID

from .batcher import Batcher
from .config import *
from .worker import Worker
from protocol.completion_task import HuggingFaceCompletionInputs, HuggingFaceCompletionOutputs
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
    def __init__(self, batcher_config: BatcherConfig, worker_config: WorkerConfig, logger: Optional[Logger] = None):
        global SERVER_SINGLETON
        if SERVER_SINGLETON is not None:
            raise ServerDoubleInitializeError()

        self.batcher_config = batcher_config
        self.worker_config = worker_config
        self.logger = logger if logger else getLogger(__name__)

        self.batcher = Batcher(config=batcher_config, logger=logger)
        self.worker = Worker(config=worker_config, logger=logger)

        self.outputs: Dict[UUID, Tuple[HuggingFaceCompletionOutputs, Optional[Error], int, float]] = dict()

        Thread(target=self._run, daemon=True).start()

        SERVER_SINGLETON = self

    def _run(self):
        while True:
            package = self.batcher.pack()
            if package is not None:
                self.outputs.update(self.worker.execute(package.prompts, package.ids, package.generation_config))
            else:
                time.sleep(self.batcher_config.packaging_interval_seconds)

    async def wait_task_done(
        self,
        inp: HuggingFaceCompletionInputs
    ) -> Tuple[HuggingFaceCompletionOutputs, Optional[Error], int, float, float]:
        uid = uuid4()
        start = time.time()

        self.logger.info(msg=f"Task-{uid} is added.")
        self.batcher.add(inp, uid)

        while True:
            await asyncio.sleep(0.1)
            if uid in self.outputs:
                end = time.time()
                outputs, error, status_code, cpu_time = self.outputs.pop(uid)
                wall_time = end - start
                self.logger.info(msg=f"Task-{uid} is finished, {cpu_time=: .4f}s, {wall_time=: .4f}s")
                return outputs, error, status_code, cpu_time, wall_time


def get_server():
    if SERVER_SINGLETON is None:
        raise ServerNotInitializedError()
    return SERVER_SINGLETON


__all__ = ["Server", "get_server"]
