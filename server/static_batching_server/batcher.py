from logging import getLogger, Logger
from typing import List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

from .config import BatcherConfig
from protocol.completion_task import HuggingFaceCompletionInputs, HuggingFaceGenerationConfig


class Package(BaseModel):
    ids: List[UUID] = Field(default=...)
    prompts: List[str] = Field(default=...)
    generation_config: HuggingFaceGenerationConfig = Field(default=...)

    def __hash__(self):
        return hash(self.generation_config)

    @property
    def workload(self):
        return len(self.prompts) * self.generation_config.num_beams

    def add(self, prompt: str, uid: UUID):
        self.prompts.append(prompt)
        self.ids.append(uid)

    def __repr__(self):
        return f"Package(workload={self.workload})"

    def __str__(self):
        return self.__repr__()


class Batcher:
    def __init__(self, config: BatcherConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger if logger else getLogger(__name__)

        self.inputs: List[Tuple[HuggingFaceCompletionInputs, UUID]] = []

    def pack(self) -> Optional[Package]:
        if not self.inputs:
            return None

        # =============================
        # Strategy 1:
        #   pack first input, then select input who has the same generation_config util package is full or
        #   there is no other input left
        # =============================

        inp, inp_id = self.inputs.pop(0)
        package = Package(ids=[inp_id], prompts=[inp.prompt], generation_config=inp.generation_config)
        inputs = []
        while self.inputs:
            if package.workload > self.config.package_max_workload:  # package is full, put back and return
                self.inputs = inputs + self.inputs
                self.logger.debug(msg=str(package))
                return package
            inp, inp_id = self.inputs.pop(0)
            if hash(inp.generation_config) != hash(package):  # gen_config is different, put back
                inputs.append((inp, inp_id))
            else:
                package.add(inp.prompt, inp_id)
        self.logger.debug(msg=str(package))
        return package

        # =============================
        # Strategy 2:
        #   pack input one by one, return immediately when package is full or generation_config is different
        # =============================

        # package = None
        # while self.inputs:
        #     inp, inp_id = self.inputs.pop(0)
        #     if package is None:  # the first input, initialize package
        #         package = Package(ids=[inp_id], prompts=[inp.prompt], generation_config=inp.generation_config)
        #     else:  # if gen_config not the same or package is full, return package, otherwise add prompt into package
        #         hash_value = hash(inp.generation_config)
        #         if hash_value != hash(package) or package.workload > self.config.package_max_workload:
        #             self.inputs.insert(0, (inp, inp_id))
        #             self.logger.debug(msg=str(package))
        #             return package
        #         package.add(inp.prompt, inp_id)
        # self.logger.debug(msg=str(package))
        # return package

    def add(self, inp: HuggingFaceCompletionInputs, inp_id: UUID):
        self.inputs.append((inp, inp_id))


__all__ = ["Batcher"]
