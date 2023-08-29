import copy
import enum
from typing import Dict, List, Optional
from uuid import uuid4, UUID

from .cache.cache import LogicalCacheBlock
from protocol.completion_task import HuggingFaceGenerationConfig


class BeamStatus(enum.Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2
    SWAPPED = 3


class BeamFinishReason(enum.Enum):
    LENGTH = "length"
    STOP = "stop"
    ABORT = "abort"
    NOT_FINISHED = "not_finished"


class Beam:
    def __init__(
        self,
        request_id: UUID,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
    ):
        self.request_id = request_id
        self.parent_beam_id = None
        self.beam_id = uuid4()
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.block_size = block_size
        self.generated_token_ids = []
        self.cumulative_logprob = 0.0

        self.cache_blocks: List[LogicalCacheBlock] = []
        self._append_tokens_to_blocks(prompt_token_ids)

        self.status = BeamStatus.WAITING
        self._finish_reason = BeamFinishReason.NOT_FINISHED

    def _append_cache_block(self):
        block = LogicalCacheBlock(block_id=len(self.cache_blocks), block_size=self.block_size)
        self.cache_blocks.append(block)

    def _append_tokens_to_blocks(self, token_ids: List[int]):
        offset = 0
        while offset < len(token_ids):
            if not self.cache_blocks:
                self._append_cache_block()

            last_block = self.cache_blocks[-1]
            if last_block.is_full:
                self._append_cache_block()
                last_block = self.cache_blocks[-1]

            num_empty_slots = last_block.num_empty_slots
            last_block.append_tokens(token_ids[offset: offset + num_empty_slots])

            offset += num_empty_slots

    def append_token_id(self, token_id: int, prob: float):
        self._append_tokens_to_blocks([token_id])
        self.generated_token_ids.append(token_id)
        self.cumulative_logprob += prob

    @property
    def last_token_id(self):
        if not self.generated_token_ids:
            return self.prompt_token_ids[-1]
        return self.generated_token_ids[-1]

    @property
    def token_ids(self):
        return self.prompt_token_ids + self.generated_token_ids

    @property
    def num_tokens(self):
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    @property
    def num_generated_tokens(self):
        return len(self.generated_token_ids)

    def update_status(self, status: BeamStatus):
        self.status = status

    def copy(self) -> "Beam":
        beam = Beam(
            request_id=self.request_id,
            prompt=self.prompt,
            prompt_token_ids=copy.deepcopy(self.prompt_token_ids),
            block_size=self.block_size
        )
        beam.parent_beam_id = self.beam_id
        beam.generated_token_ids = copy.deepcopy(self.generated_token_ids)
        beam.cumulative_logprob = self.cumulative_logprob

        beam.cache_blocks = copy.deepcopy(self.cache_blocks)
        beam.status = copy.deepcopy(self.status)
        beam._finish_reason = copy.deepcopy(self._finish_reason)

        return beam

    def check_finished(
        self,
        eos_token_id: int,
        max_new_tokens: int
    ) -> None:
        if not self.generated_token_ids:
            return
        if eos_token_id == int(self.generated_token_ids[-1]):
            self.status = BeamStatus.FINISHED
            self.finish_reason = BeamFinishReason.STOP
        if len(self.generated_token_ids) == max_new_tokens:
            self.status = BeamStatus.FINISHED
            self.finish_reason = BeamFinishReason.LENGTH

    def __eq__(self, other: "Beam"):
        return self.cumulative_logprob == other.cumulative_logprob

    def __gt__(self, other: "Beam"):
        return self.cumulative_logprob > other.cumulative_logprob

    def __ge__(self, other: "Beam"):
        return self.cumulative_logprob >= other.cumulative_logprob

    def __lt__(self, other: "Beam"):
        return self.cumulative_logprob < other.cumulative_logprob

    def __le__(self, other: "Beam"):
        return self.cumulative_logprob <= other.cumulative_logprob

    @property
    def finish_reason(self) -> str:
        return self._finish_reason.value

    @finish_reason.setter
    def finish_reason(self, finish_reason: BeamFinishReason):
        self._finish_reason = finish_reason

    @property
    def is_finished(self) -> bool:
        return self.status == BeamStatus.FINISHED


class BeamGroup:
    """ A group of beams that are generated from the same prompt"""
    def __init__(
        self,
        request_id: UUID,
        arrival_time: float,
        beams: List[Beam],
        generation_config: HuggingFaceGenerationConfig
    ):
        self.request_id = request_id
        self.arrival_time = arrival_time
        self.beams: Dict[UUID, Beam] = {beam.beam_id: beam for beam in beams}
        self.generation_config = generation_config

        self._new_beams: List[Beam] = []

    def add_beams(self, beams: List[Beam]):
        for beam in beams:
            self.beams[beam.beam_id] = beam

    def get_beams(self, status: Optional[BeamStatus] = None) -> List[Beam]:
        if status is None:
            return list(self.beams.values())
        else:
            return [beam for beam in self.beams.values() if beam.status == status]

    def num_beams(self, status: Optional[BeamStatus] = None) -> int:
        return len(self.get_beams(status))

    def cache_new_beams(self, new_beams: List[Beam]):
        self._new_beams += new_beams

    def clear_new_beams(self):
        self._new_beams = []

    @property
    def new_beams(self):
        return self._new_beams

    @staticmethod
    def update_beam_status(beam: Beam, status: BeamStatus):
        beam.update_status(status)

    def find(self, beam_id: UUID) -> Beam:
        if beam_id not in self.beams:
            raise LookupError(f"Beam {beam_id} not found.")
        return self.beams[beam_id]

    @property
    def is_finished(self) -> bool:
        if not self.generation_config.early_stopping:
            return all(beam.is_finished for beam in self.beams.values())
        else:
            num_finished_beams = len(self.get_beams(BeamStatus.FINISHED))
            return num_finished_beams == self.generation_config.num_beams

    def get_final_beams(self) -> List[Beam]:
        if not self.is_finished:
            raise AttributeError("Can't get final beams for they are not finished.")
        return sorted(self.get_beams(BeamStatus.FINISHED), reverse=True)[:self.generation_config.num_return_sequences]


__all__ = [
    "Beam",
    "BeamGroup",
    "BeamStatus",
    "BeamFinishReason",
]
