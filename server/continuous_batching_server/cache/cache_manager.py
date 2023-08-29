# adopt from vllm

from logging import getLogger
from typing import *
from uuid import UUID

from .cache import LogicalCacheBlock, PhysicalCacheBlock
from ..beam import Beam, BeamGroup, BeamStatus


logger = getLogger(__name__)


class OOMError(Exception):
    pass


class DoubleFreeBlockError(Exception):
    pass


class CacheBlockAllocator:
    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.free_blocks: List[PhysicalCacheBlock] = [
            PhysicalCacheBlock(block_id=i, block_size=block_size) for i in range(num_blocks)
        ]

    def allocate(self) -> PhysicalCacheBlock:
        if not self.free_blocks:
            raise OOMError("No free blocks are available.")
        block = self.free_blocks.pop(0)
        block.ref_count = 1
        return block

    def free(self, block: PhysicalCacheBlock) -> None:
        if block.ref_count == 0:
            raise DoubleFreeBlockError(f"Double free! block-{block.block_id} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)
            # self.free_blocks = sorted(self.free_blocks, key=lambda block: block.block_id)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)


BlockTable = List[PhysicalCacheBlock]


class CacheBlockManager:
    def __init__(self, block_size: int, num_blocks: int, num_blocks_cpu: int, watermark: float = 0.01):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.watermark = watermark
        assert self.watermark >= 0.0

        self.watermark_blocks = int(watermark * num_blocks)
        self.allocator = CacheBlockAllocator(block_size, num_blocks)
        self.cpu_allocator = CacheBlockAllocator(block_size, num_blocks_cpu)

        self.block_tables: Dict[UUID, BlockTable] = {}

    def can_allocate(self, beam: Beam):
        num_required_blocks = len(beam.cache_blocks)
        num_free_blocks = self.allocator.num_free_blocks
        # Use watermark to avoid frequent cache eviction.
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def allocate(self, beam: Beam):
        # NOTE: only do to beam that is 'init' beam.
        block_table: BlockTable = []

        # Allocate new physical cache blocks that will store the prompt tokens.
        for _ in range(len(beam.cache_blocks)):
            block = self.allocator.allocate()
            block_table.append(block)

        self.block_tables[beam.beam_id] = block_table.copy()
        logger.debug(f"beam-{beam.beam_id} allocate block_table: {[block.block_id for block in block_table]}")

    def is_need_to_append_slot(self, beam: Beam):
        logical_blocks = beam.cache_blocks
        block_table: BlockTable = self.block_tables[beam.beam_id]

        if len(block_table) < len(logical_blocks):
            return True
        if block_table[-1].ref_count > 1:
            # The last block is shared with other beams, which means should copy on write
            return True
        return False

    def can_append_slot(self):
        return self.allocator.num_free_blocks >= 1

    def append_slot(self, beam: Beam) -> Optional[Tuple[int, int]]:
        logical_blocks = beam.cache_blocks
        block_table: BlockTable = self.block_tables[beam.beam_id]

        if len(block_table) < len(logical_blocks):
            block = self.allocator.allocate()
            block_table.append(block)
            logger.debug(f"beam-{beam.beam_id} add one block-{block.block_id}")
            return

        last_block = block_table[-1]
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self.allocator.allocate()
            block_table[-1] = new_block
            self.allocator.free(last_block)
            logger.debug(f"beam-{beam.beam_id} replace block-{last_block.block_id} with block-{new_block.block_id}")
            return last_block.block_id, new_block.block_id

    def copy(self, parent: UUID, child: UUID):
        src_block_table = self.block_tables[parent]
        self.block_tables[child] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1
        logger.debug(f"beam-{parent} copy block_table: {[block.block_id for block in src_block_table]} to beam-{child}")

    def _get_physical_blocks(self, beam_group: BeamGroup) -> List[PhysicalCacheBlock]:
        blocks: Set[PhysicalCacheBlock] = set()
        for beam in beam_group.get_beams():
            if beam.is_finished:
                continue
            block_table = self.block_tables[beam.beam_id]
            for block in block_table:
                blocks.add(block)
        return list(blocks)

    def can_swap_in(self, beam_group: BeamGroup, preserved_num_blocks: int = 0) -> bool:
        blocks = self._get_physical_blocks(beam_group)
        num_swapped_seqs = beam_group.num_beams(status=BeamStatus.SWAPPED)
        num_free_blocks = self.allocator.num_free_blocks
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks - preserved_num_blocks >= self.watermark_blocks

    def swap_in(self, beam_group: BeamGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        mapping: Dict[PhysicalCacheBlock, PhysicalCacheBlock] = {}
        for beam in beam_group.get_beams():
            if beam.is_finished:
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[beam.beam_id]

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.allocator.allocate()
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
            self.block_tables[beam.beam_id] = new_block_table

        block_ids_map = {
            cpu_block.block_id: gpu_block.block_id
            for cpu_block, gpu_block in mapping.items()
        }
        logger.debug(
            f"swap in beam_group-{beam_group.request_id} from CPU to GPU."
        )
        return block_ids_map

    def can_swap_out(self, beam_group: BeamGroup) -> bool:
        blocks = self._get_physical_blocks(beam_group)
        return len(blocks) <= self.cpu_allocator.num_free_blocks

    def swap_out(self, beam_group: BeamGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalCacheBlock, PhysicalCacheBlock] = {}
        for beam in beam_group.get_beams():
            if beam.is_finished:
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[beam.beam_id]

            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.allocator.free(gpu_block)
            self.block_tables[beam.beam_id] = new_block_table

        block_ids_map = {
            gpu_block.block_id: cpu_block.block_id
            for gpu_block, cpu_block in mapping.items()
        }
        logger.debug(
            f"swap out beam_group-{beam_group.request_id} from GPU to CPU."
        )
        return block_ids_map

    def _free_block_table(self, block_table: BlockTable):
        for block in block_table:
            self.allocator.free(block)

    def free(self, beam_id: UUID):
        if beam_id not in self.block_tables:
            return
        block_table = self.block_tables.pop(beam_id)
        self._free_block_table(block_table)
        logger.debug(f"beam-{beam_id} free blocks: {[block.block_id for block in block_table]}")

    def reset(self):
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, beam_id: UUID) -> List[int]:
        block_table = self.block_tables[beam_id]
        return [block.block_id for block in block_table]

    @property
    def num_free_blocks(self):
        return self.allocator.num_free_blocks

    @property
    def num_free_cpu_blocks(self):
        return self.cpu_allocator.num_free_blocks
