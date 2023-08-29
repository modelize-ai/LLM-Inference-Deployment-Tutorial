from collections import defaultdict
from logging import getLogger, Logger
from typing import *
from uuid import UUID

import torch

from .beam import Beam, BeamGroup, BeamStatus
from .config import BatcherConfig, CacheConfig
from .cache.cache_manager import CacheBlockManager
from .generation_utils import HeterogeneousNextTokensChooser


class Batch:
    def __init__(self):
        self.prefill_beams: List[Beam] = []
        self.generation_beams: List[Beam] = []
        self.block_tables: Dict[UUID, List[int]] = {}
        self.prefill_logits: Optional[torch.Tensor] = None
        self.generation_logits: Optional[torch.Tensor] = None

    @property
    def num_beams(self) -> int:
        return len(self.prefill_beams) + len(self.generation_beams)

    @property
    def request_ids(self):
        return list(set([beam.request_id for beam in self.prefill_beams + self.generation_beams]))

    def add(self, beam: Beam, block_ids: List[int]) -> None:
        if beam.num_generated_tokens == 0:
            self.prefill_beams.append(beam)
        else:
            self.generation_beams.append(beam)
        self.block_tables[beam.beam_id] = block_ids


class Batcher:
    def __init__(
        self,
        batcher_config: BatcherConfig,
        cache_config: CacheConfig,
        logger: Optional[Logger] = None
    ):
        self.batcher_config = batcher_config
        self.cache_config = cache_config

        self.logger = logger if logger else getLogger(__name__)

        self.waiting: List[BeamGroup] = []
        self.running: List[BeamGroup] = []
        self.preempting: List[BeamGroup] = []

        self.req_id2beam_group: Dict[UUID, BeamGroup] = {}

        self.cache_manager = CacheBlockManager(
            cache_config.block_size,
            cache_config.num_blocks,
            cache_config.num_blocks_cpu,
            cache_config.watermark
        )

    def add_request(self, request: BeamGroup):
        self.waiting.append(request)
        self.req_id2beam_group[request.request_id] = request

    def _allocate(self, beam: Beam) -> bool:
        if not self.cache_manager.can_allocate(beam):
            return False
        self.cache_manager.allocate(beam)
        return True

    def _append_slot(self, beam: Beam, blocks_to_copy: Dict[int, List[int]]):
        ret = self.cache_manager.append_slot(beam)
        if ret is not None:
            src_block, dst_block = ret
            blocks_to_copy[src_block].append(dst_block)

    def _free_finished_beams(self):
        for beam in [beam for beam_group in self.running for beam in beam_group.get_beams(BeamStatus.FINISHED)]:
            self.logger.debug(f"free blocks of beam-{beam.beam_id} for it is finished.")
            self.cache_manager.free(beam.beam_id)

    def schedule(self) -> Tuple[
        Batch,
        Dict[int, List[int]],
        Dict[int, int],
        Dict[int, int],
        List[BeamGroup]
    ]:
        """
        大致的执行流程如下：
        1. 从运行队列中移除已经完成生成的请求（beam_group）
        2. 对运行队列中仍需继续生成的请求进行缓存空间的分配
            a. 统计运行队列中所有请求所需增量分配的 GPU 缓存空间的总块数
            b. 如果空闲的 GPU 缓存空间总块数少于增量分配所需的总块数，则从运行队列尾端起依次将每个请求所占用的 GPU 缓存空间交互至 CPU，直至
                剩余空间足够用于分配给运行队列前端的其他请求
            c. 如果不发生 GPU->CPU 交互且存在被交换至 CPU 的等待继续被处理的请求时，将这些请求按优先级依次交换回 GPU 直至无法被换回
            d. 为运行队列中剩余的请求进行缓存空间的分配
        3. 在不发生 GPU->CPU 交互的情况下，或运行队列无请求时，尝试将等待队列中的请求移至运行队列

        :return: (
            Batch,
            blocks_to_copy: Dict[int, List[int]],
            blocks_to_swap_in: Dict[int, int],
            blocks_to_swap_out: Dict[int, int],
            finished_requests: List[BeamGroup]
        )
        """
        batch = Batch()

        # step1: 从运行队列中移除已经完成生成的请求（beam_group）
        self._free_finished_beams()
        running = []
        finishing = []
        while self.running:
            beam_group = self.running.pop(0)
            if beam_group.is_finished:
                finishing.append(beam_group)
            else:
                running.append(beam_group)
        self.running = running

        # step2: 对运行队列中仍需继续生成的请求进行缓存空间的分配
        running = []
        swapping_out = []
        swapping_in = []
        blocks_to_copy: Dict[int, List[int]] = defaultdict(list)
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        # step2.a: 统计运行队列中所有请求所需增量分配的 GPU 缓存空间的总块数
        run_request2num_append_blocks = defaultdict(int)
        for beam_group in self.running:
            run_request2num_append_blocks[beam_group.request_id] = 0
            for beam in beam_group.get_beams(BeamStatus.RUNNING):
                if self.cache_manager.is_need_to_append_slot(beam):
                    run_request2num_append_blocks[beam_group.request_id] += 1
        # step2.b: 如果空闲的 GPU 缓存空间总块数少于增量分配所需的总块数，则从运行队列尾端起
        #          依次将每个请求所占用的 GPU 缓存空间交互至 CPU，
        #          直至剩余空间足够用于分配给运行队列前端的其他请求
        while self.cache_manager.allocator.num_free_blocks < sum(run_request2num_append_blocks.values()):
            beam_group = self.running.pop(-1)
            num_append_blocks = run_request2num_append_blocks.pop(beam_group.request_id)
            if num_append_blocks == 0:
                running.insert(0, beam_group)
                continue
            if not self.cache_manager.can_swap_out(beam_group):
                # FIXME: do not raise error, abort this beam group, mark as finished with an abortion reason, free cache space
                raise RuntimeError("No enough CPU RAM to swap out")
            else:
                blocks_to_swap_out.update(self.cache_manager.swap_out(beam_group))
                for beam in beam_group.get_beams(BeamStatus.RUNNING):
                    beam_group.update_beam_status(beam, BeamStatus.SWAPPED)
                swapping_out.insert(0, beam_group)
                self.logger.debug(
                    f"one request swapped out, "
                    f"free_gpu_blocks={self.cache_manager.allocator.num_free_blocks}, "
                    f"free_cpu_blocks={self.cache_manager.cpu_allocator.num_free_blocks}"
                )
        self.running += running
        self.preempting += swapping_out
        # step2.c: 如果不发生 GPU->CPU 交互且存在被交换至 CPU 的等待继续被处理的请求时，
        #          将这些请求按优先级依次交换回 GPU 直至无法被换回
        if not swapping_out:
            preserved_num_blocks = sum(run_request2num_append_blocks.values())
            while self.preempting:
                beam_group = self.preempting[0]
                if not self.cache_manager.can_swap_in(beam_group, preserved_num_blocks):
                    self.logger.debug(
                        f"attempt to swap in beam_group-{beam_group.request_id} but not have enough free gpu blocks."
                    )
                    if not self.running:
                        raise RuntimeError(
                            "running queue is empty but still can't swap in request, "
                            "please consider increase num_blocks or decrease max tokens number"
                        )
                    else:
                        break  # exceed num available free gpu blocks if swap in this beam_group, break
                beam_group = self.preempting.pop(0)
                blocks_to_swap_in.update(self.cache_manager.swap_in(beam_group))
                for beam in beam_group.get_beams(BeamStatus.SWAPPED):
                    beam_group.update_beam_status(beam, BeamStatus.RUNNING)
                swapping_in.append(beam_group)
                preserved_num_blocks += sum(
                    [
                        self.cache_manager.is_need_to_append_slot(beam)
                        for beam in beam_group.get_beams(BeamStatus.RUNNING)
                    ]
                )
                self.logger.debug(
                    f"one request swapped in, "
                    f"free_gpu_blocks={self.cache_manager.allocator.num_free_blocks}, "
                    f"free_cpu_blocks={self.cache_manager.cpu_allocator.num_free_blocks}"
                )
            self.running += swapping_in
        # step2.d: 为运行队列中剩余的请求进行缓存空间的分配
        for beam_group in self.running:
            self.logger.debug(
                f"beam_group-{beam_group.request_id}'s beams' status: "
                f"{[beam.status.name for beam in beam_group.get_beams()]}"
            )
            beams = beam_group.get_beams(BeamStatus.RUNNING)
            for beam in beams:
                self._append_slot(beam, blocks_to_copy)
                block_ids = self.cache_manager.get_block_table(beam.beam_id)
                batch.add(beam, block_ids)
                beam_group.beams.pop(beam.beam_id)

        # step3. 在不发生 GPU->CPU 交互的情况下，尝试将等待队列中的请求移至运行队列
        batch_tokens = batch.num_beams
        if (not swapping_out or not self.running) and not self.preempting:
            while self.waiting:
                beam_group = self.waiting[0]
                beam = beam_group.get_beams()[0]
                if batch_tokens + beam.num_tokens > self.batcher_config.batch_max_tokens:
                    self.logger.debug(
                        f"reach batch_max_tokens {self.batcher_config.batch_max_tokens}, "
                        f"current batch_tokens {batch_tokens}"
                    )
                    break
                if batch.num_beams + 1 > self.batcher_config.batch_max_beams:
                    self.logger.debug(
                        f"reach batch_max_beams {self.batcher_config.batch_max_beams}, "
                        f"current batch_beams {batch.num_beams}"
                    )
                    break
                has_cache_space = self._allocate(beam)
                if not has_cache_space:
                    self.logger.debug("hasn't cache space to allocate")
                    break
                beam_group = self.waiting.pop(0)
                beam = beam_group.get_beams()[0]
                batch_tokens += beam.num_tokens
                beam_group.update_beam_status(beam, BeamStatus.RUNNING)
                self.running.append(beam_group)
                block_ids = self.cache_manager.get_block_table(beam.beam_id)
                batch.add(beam, block_ids)
                beam_group.beams.pop(beam.beam_id)

        return batch, blocks_to_copy, blocks_to_swap_in, blocks_to_swap_out, finishing

    def _generate(self, old_beams: List[Beam], logits: torch.Tensor):
        next_tokens_chooser = HeterogeneousNextTokensChooser(
            dtype=logits.dtype,
            device=logits.device,
            temperature=[self.req_id2beam_group[beam.request_id].generation_config.temperature for beam in old_beams],
            repetition_penalty=[self.req_id2beam_group[beam.request_id].generation_config.repetition_penalty for beam in old_beams],
            top_k=[self.req_id2beam_group[beam.request_id].generation_config.top_k for beam in old_beams],
            top_p=[self.req_id2beam_group[beam.request_id].generation_config.top_p for beam in old_beams],
            typical_p=[self.req_id2beam_group[beam.request_id].generation_config.typical_p for beam in old_beams],
            do_sample=[self.req_id2beam_group[beam.request_id].generation_config.do_sample for beam in old_beams],
            num_beams=[self.req_id2beam_group[beam.request_id].generation_config.num_beams for beam in old_beams],
            seeds=[self.req_id2beam_group[beam.request_id].generation_config.seed for beam in old_beams]
        )
        all_input_ids = [beam.token_ids for beam in old_beams]
        max_input_len = max([len(input_ids) for input_ids in all_input_ids])
        all_input_ids = [input_ids + [0] * (max_input_len - len(input_ids)) for input_ids in all_input_ids]
        all_input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long, device=logits.device)
        outputs = next_tokens_chooser(
            input_ids=all_input_ids_tensor,
            scores=logits
        )
        for beam, output in zip(old_beams, outputs):
            new_beams = []
            generation_config = self.req_id2beam_group[beam.request_id].generation_config
            for nxt_token_id, nxt_prob in zip(output.next_token_ids, output.next_probs):
                new_beam = beam.copy()
                new_beam.append_token_id(nxt_token_id, nxt_prob)
                new_beam.check_finished(
                    eos_token_id=generation_config.eos_token_id,
                    max_new_tokens=generation_config.max_new_tokens
                )
                new_beams.append(new_beam)
            self.req_id2beam_group[beam.request_id].cache_new_beams(new_beams)

        req_ids = list(set([beam.request_id for beam in old_beams]))
        for req_id in req_ids:
            beam_group = self.req_id2beam_group[req_id]
            if not beam_group.get_beams(BeamStatus.RUNNING):
                new_beams = beam_group.new_beams
                beam_group.clear_new_beams()
                new_beams = sorted(new_beams, reverse=True)[:beam_group.generation_config.num_beams]
                beam_group.add_beams(new_beams)
                for new_beam in new_beams:
                    if not new_beam.is_finished:
                        self.cache_manager.copy(new_beam.parent_beam_id, new_beam.beam_id)
                for old_beam in old_beams:
                    if old_beam.request_id == req_id:
                        self.cache_manager.free(old_beam.beam_id)

    def batch_generate(self, batch: Batch):
        if batch.prefill_beams:
            self._generate(batch.prefill_beams, batch.prefill_logits)
        if batch.generation_beams:
            self._generate(batch.generation_beams, batch.generation_logits)


__all__ = ["Batch", "Batcher"]
