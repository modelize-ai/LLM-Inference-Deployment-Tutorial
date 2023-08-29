import time
from logging import getLogger, Logger
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

from .config import WorkerConfig
from protocol.completion_task import (
    TokenUsage,
    HuggingFaceGenerationConfig,
    HuggingFaceCompletionChoice,
    HuggingFaceCompletionOutputs
)
from protocol.error import Error


class TextGenerationPipeline:
    """A simplified pipeline to show what HF's TextGenerationPipeline mainly do under the hood"""
    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, batch_size: int = -1):
        self.model = model
        self.tokenizer = tokenizer
        device = model.device
        if model.hf_device_map:
            device = model.hf_device_map[next(iter(model.hf_device_map))]
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.batch_size = batch_size

    def _preprocess(
        self,
        prompt_texts: List[str],
        generation_config: HuggingFaceGenerationConfig,
        handle_long_generation=None,
    ) -> "BatchEncoding":
        inputs = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            new_tokens = generation_config.max_new_tokens
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the"
                        " models max length"
                    )

                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -keep_length:]

        return inputs

    def _model_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: HuggingFaceGenerationConfig
    ) -> torch.Tensor:
        batch_gen_sequences = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_config.dict(by_alias=True)
        )
        return batch_gen_sequences

    def _postprocess(
        self,
        input_ids: torch.Tensor,
        generated_sequences: torch.Tensor,
        clean_up_tokenization_spaces=True
    ) -> List[HuggingFaceCompletionOutputs]:
        input_ids = input_ids.cpu()
        generated_sequences = generated_sequences.cpu()

        num_return_sequences = len(generated_sequences) // len(input_ids)
        batch_outputs = []
        for idx, start in enumerate(range(0, len(generated_sequences), num_return_sequences)):
            inp = input_ids[idx].tolist()
            if self.tokenizer.pad_token_id in inp:
                inp = inp[:inp.index(self.tokenizer.pad_token_id)]

            sequences = generated_sequences[start: start + num_return_sequences]
            sequences = sequences[..., input_ids[idx].size(0):].tolist()
            for i, seq in enumerate(sequences):
                if self.tokenizer.pad_token_id in seq:
                    sequences[i] = seq[: seq.index(self.tokenizer.pad_token_id)]

            usage = TokenUsage(prompt_tokens=len(inp), completion_tokens=sum([len(seq) for seq in sequences]))
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

            generated_texts = self.tokenizer.batch_decode(
                sequences,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                skip_special_tokens=True
            )
            choices = [
                HuggingFaceCompletionChoice(
                    text=text,
                    index=index,
                    finish_reason="stop" if self.tokenizer.eos_token in text else "length"
                )
                for index, text in enumerate(generated_texts)
            ]

            batch_outputs.append(HuggingFaceCompletionOutputs(choices=choices, usage=usage))

        return batch_outputs

    def __call__(
        self,
        text_inputs,
        generation_config: HuggingFaceGenerationConfig,
        clean_up_tokenization_spaces=False,
        handle_long_generation="hole",
    ) -> List[HuggingFaceCompletionOutputs]:
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        outputs = []
        batch_size = self.batch_size
        if batch_size == -1:
            batch_size = len(text_inputs)
        for start in range(0, len(text_inputs), batch_size):
            batch_input_texts = text_inputs[start: start + batch_size]

            batch_inputs = self._preprocess(batch_input_texts, generation_config, handle_long_generation)
            batch_input_ids = batch_inputs.input_ids
            batch_attention_mask = batch_inputs.attention_mask

            batch_gen_sequences = self._model_generate(batch_input_ids, batch_attention_mask, generation_config)

            outputs += self._postprocess(batch_input_ids, batch_gen_sequences, clean_up_tokenization_spaces)
        return outputs


class Worker:
    def __init__(self, config: WorkerConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger if logger else getLogger(__name__)

        self.model, self.tokenizer = self._load_model_tokenizer()
        self.pipeline = TextGenerationPipeline(self.model, self.tokenizer, batch_size=self.config.batch_size)

    def _load_model_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name_or_path or self.config.model_name_or_path,
            use_fast=self.config.use_fast_tokenizer,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.config.trust_remote_code
        )
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        max_memory = self.config.max_memory
        if max_memory:
            max_memory = {(eval(k) if isinstance(k, str) else k): v for k, v in max_memory.items()}

        if self.config.is_gptq_quantized:
            model = AutoGPTQForCausalLM.from_quantized(
                model_name_or_path=self.config.model_name_or_path,
                device_map=self.config.device_map,
                max_memory=max_memory,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                trust_remote_code=self.config.trust_remote_code,
                use_safetensors=self.config.use_safetensors
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config.model_name_or_path,
                torch_dtype=getattr(torch.dtype, self.config.torch_dtype),
                device_map=self.config.device_map,
                max_memory=max_memory,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                revision=self.config.revision,
                trust_remote_code=self.config.trust_remote_code
            )

        return model, tokenizer

    def execute(
        self,
        prompts: List[str],
        uids: List[UUID],
        generation_config: HuggingFaceGenerationConfig
    ) -> Dict[UUID, Tuple[HuggingFaceCompletionOutputs, Optional[Error], int, float]]:
        start = time.time()
        try:
            pipeline_results = self.pipeline(prompts, generation_config)
            end = time.time()
            return {uid: (outputs, None, 200, end - start) for uid, outputs in zip(uids, pipeline_results)}
        except Exception as e:
            end = time.time()
            error = Error(type=e.__class__.__name__, detail=str(e))
            self.logger.error(msg=str(error), exc_info=e)
            return {uid: (HuggingFaceCompletionOutputs(), error, 500, end - start) for uid in uids}


__all__ = ["Worker"]
