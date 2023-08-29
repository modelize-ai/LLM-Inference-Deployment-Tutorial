import json
import os
from logging import getLogger, Logger
from typing import *

import torch
from auto_gptq import BaseQuantizeConfig as GPTQConfig
from auto_gptq.modeling._utils import autogptq_post_init

from .batcher import Batch
from .cache.cache import Cache
from .modeling.utils.weights import Weights
from .config import (
    CacheConfig,
    ModelConfig,
    ModelLoadingConfig,
    MODEL_AUTO_TABLE,
    ParallelConfig,
    TORCH_FLOAT_DTYPE_MAP
)


def _get_gptq_config(config_dir: str, config_file_base_name: Optional[str] = None) -> GPTQConfig:
    config_path = os.path.join(config_dir, "quantize_config.json")
    if config_file_base_name:
        config_path = os.path.join(config_dir, f"{config_file_base_name}.json")
    return GPTQConfig(**json.load(open(config_path, "r", encoding="utf-8")))


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))


class Worker:
    def __init__(
        self,
        cache_config: CacheConfig,
        model_loading_config: ModelLoadingConfig,
        parallel_config: ParallelConfig,
        logger: Optional[Logger] = None
    ):
        self.cache_config = cache_config
        self.model_loading_config = model_loading_config
        self.parallel_config = parallel_config

        self.logger = logger if logger else getLogger(__name__)

        # load model
        self.device = torch.device(self.model_loading_config.device)
        self.dtype = TORCH_FLOAT_DTYPE_MAP[self.model_loading_config.torch_dtype]

        torch.cuda.set_device(self.device)

        factory = MODEL_AUTO_TABLE[self.model_loading_config.model_type]
        model_config_cls = factory.model_config_cls
        model_cls = factory.model_cls
        model_config = model_config_cls.from_pretrained(
            self.model_loading_config.model_name_or_path,
            trust_remote_code=self.model_loading_config.trust_remote_code
        )
        model_weights = Weights(
            self.model_loading_config.model_name_or_path,
            device=self.device,
            dtype=self.dtype,
            quantize_method=self.model_loading_config.quantize_method,
            gptq_model_base_name=self.model_loading_config.gptq_model_base_name
        )
        if self.model_loading_config.quantize_method == "gptq":
            gptq_config = _get_gptq_config(
                self.model_loading_config.model_name_or_path,
                self.model_loading_config.gptq_config_base_name
            )
        else:
            gptq_config = None
        self.model = model_cls(config=model_config, weights=model_weights, gptq_config=gptq_config)
        if self.model_loading_config.quantize_method == "gptq":
            self.model = autogptq_post_init(self.model, gptq_config.desc_act)
        self.model.eval()
        self.model_config = ModelConfig(self.model.config, self.parallel_config)

        self.cache = Cache(
            cache_config=self.cache_config,
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            dtype=self.dtype,
            device=self.device
        )

    def _prepare_inputs(self, batch: Batch, prefill: bool) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        input_ids: List[int] = []
        position_ids: List[int] = []
        block_tables: Optional[List[List[int]]] = []
        slots: List[int] = []
        context_lengths: List[int] = []
        lm_head_indices: Optional[List[int]] = []

        beams = batch.prefill_beams if prefill else batch.generation_beams
        if not beams:
            return
        if prefill:
            block_tables = None
            for beam in beams:
                input_ids += beam.prompt_token_ids
                position_ids += list(range(beam.num_tokens))
                context_lengths.append(beam.num_tokens)
                lm_head_indices.append(sum(context_lengths) - 1)

                block_ids = batch.block_tables[beam.beam_id]
                if block_ids is None:
                    slots += [0] * beam.num_tokens
                else:
                    for i in range(beam.num_tokens):
                        block_id = block_ids[i // self.cache.block_size]
                        block_offset = i % self.cache.block_size
                        slot = block_id * self.cache.block_size + block_offset
                        slots.append(slot)
        else:
            lm_head_indices = None
            for beam in beams:
                input_ids.append(beam.last_token_id)
                position_ids.append(beam.num_tokens - 1)
                context_lengths.append(beam.num_tokens)

                block_ids = batch.block_tables[beam.beam_id]
                block_tables.append(block_ids)

                block_id = block_ids[position_ids[-1] // self.cache.block_size]
                block_offset = position_ids[-1] % self.cache_config.block_size
                slot = block_id * self.cache_config.block_size + block_offset
                slots.append(slot)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        # FIXME: after padding, model execution will fail if bsz is not a multiple of 8
        # input_ids = _pad_to_alignment(input_ids, multiple_of=8)
        # position_ids = _pad_to_alignment(position_ids, multiple_of=8)

        # Convert to tensors.
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        position_ids = torch.tensor(position_ids, dtype=torch.long, device=self.device)
        context_lengths = torch.tensor(context_lengths, dtype=torch.int32, device=self.device)
        if block_tables is not None:
            max_num_blocks = max([len(block_table) for block_table in block_tables])
            block_tables = torch.IntTensor(
                [_pad_to_max(block_table, max_num_blocks) for block_table in block_tables]
            ).to(self.device)
        slots = torch.IntTensor(slots).to(self.device)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "kv_cache": self.cache.cache,
            "prefill": prefill,
            "block_tables": block_tables,
            "slots": slots,
            "context_lengths": context_lengths,
            "lm_head_indices": lm_head_indices
        }

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def _forward(self, batch: Batch, cache_events: Optional[List[torch.cuda.Event]] = None):
        prefill_inputs = self._prepare_inputs(batch, prefill=True)
        generation_inputs = self._prepare_inputs(batch, prefill=False)

        if prefill_inputs:
            self.logger.debug("executing model for prefilling.")
            batch.prefill_logits = self.model(cache_events=cache_events, **prefill_inputs)
            self.logger.debug("executed model for prefilling.")
        if generation_inputs:
            self.logger.debug("executing model for decoding.")
            batch.generation_logits = self.model(cache_events=cache_events, **generation_inputs)
            self.logger.debug("executed model for decoding.")

    def forward(
        self,
        batch: Batch,
        blocks_to_copy: Dict[int, List[int]],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int]
    ):
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.logger.debug("executing cache swap in operation.")
            self.cache.swap_in(blocks_to_swap_in)
            issued_cache_op = True
            self.logger.debug("executed cache swap in operation.")
        if blocks_to_swap_out:
            self.logger.debug("executing cache swap out operation.")
            self.cache.swap_out(blocks_to_swap_out)
            issued_cache_op = True
            self.logger.debug("executed cache swap out operation.")
        if blocks_to_copy:
            self.logger.debug("execution cache copy operation.")
            self.cache.copy(blocks_to_copy)
            issued_cache_op = True
            self.logger.debug("executed cache copy operation")

        if issued_cache_op:
            cache_events = self.cache.events
        else:
            cache_events = None

        if batch.num_beams == 0:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            self.logger.debug("no beams need to be processed, return directly.")
            return

        self._forward(batch, cache_events=cache_events)


__all__ = ["Worker"]
