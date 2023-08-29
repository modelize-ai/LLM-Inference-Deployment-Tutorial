# adopt from vllm

from typing import *

import torch
import vllm.cache_ops as vllm_cache_ops

from ..config import CacheConfig, ModelConfig, ParallelConfig


KVCache = Tuple[torch.Tensor, torch.Tensor]


class LogicalCacheBlock:
    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size

        self._token_ids = [-1] * block_size
        self.num_tokens = 0

    @property
    def is_empty(self):
        return self.num_tokens == 0

    @property
    def num_empty_slots(self):
        return self.block_size - self.num_tokens

    @property
    def is_full(self):
        return self.num_tokens == self.block_size

    def append_tokens(self, token_ids: List[int]):
        assert len(token_ids) <= self.num_empty_slots
        offset = self.num_tokens
        self._token_ids[offset: offset + len(token_ids)] = token_ids
        self.num_tokens += len(token_ids)

    @property
    def token_ids(self):
        return self._token_ids[:self.num_tokens]

    @property
    def last_token_id(self):
        assert self.num_tokens > 0
        return self.token_ids[self.num_tokens - 1]


class PhysicalCacheBlock:
    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size

        self.ref_count = 0

    def __repr__(self):
        return f"[block_id={self.block_id} block_size={self.block_size} ref_count={self.ref_count}]"

    def __str__(self):
        return self.__repr__()


class Cache:
    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        dtype: torch.dtype,
        device: torch.device
    ):
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers()
        self.num_heads = model_config.get_num_heads()

        self.dtype = dtype
        self.device = device

        self.block_size = cache_config.block_size
        self.num_blocks = cache_config.num_blocks
        self.num_blocks_cpu = cache_config.num_blocks_cpu

        self.cache = [
            (
                torch.empty(
                    size=(self.num_blocks, *self.key_block_shape),
                    dtype=self.dtype,
                    device=self.device
                ),
                torch.empty(
                    size=(self.num_blocks, *self.value_block_shape),
                    dtype=self.dtype,
                    device=self.device
                )
            ) for _ in range(self.num_layers)
        ]

        self.cpu_cache = [
            (
                torch.empty(
                    size=(self.num_blocks_cpu, *self.key_block_shape),
                    dtype=self.dtype,
                    pin_memory=cache_config.pin_memory_on_cpu,
                ),
                torch.empty(
                    size=(self.num_blocks_cpu, *self.value_block_shape),
                    dtype=self.dtype,
                    pin_memory=cache_config.pin_memory_on_cpu,
                )
            ) for _ in range(self.num_layers)
        ]

        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()

        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    @property
    def key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    @property
    def value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                vllm_cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                vllm_cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.cache]
        value_caches = [value_cache for _, value_cache in self.cache]
        # This operation implicitly synchronizes the CPU and GPU.
        vllm_cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        dtype: torch.dtype
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_heads()
        num_layers = model_config.get_num_layers()

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        return dtype_size * total


__all__ = [
    "LogicalCacheBlock",
    "PhysicalCacheBlock",
    "Cache"
]
