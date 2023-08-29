from collections import namedtuple
from typing import Optional

import torch
from pydantic import BaseModel, Field
from transformers import PretrainedConfig

from .modeling.llama import LlamaForCausalLM, LlamaConfig


ModelFactory = namedtuple(
    "ModelFactory",
    [
        "model_cls",
        "model_config_cls"
    ]
)


TORCH_FLOAT_DTYPE_MAP = {
    "float": torch.float32,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64
}

MODEL_AUTO_TABLE = {
    "llama": ModelFactory(model_cls=LlamaForCausalLM, model_config_cls=LlamaConfig)
}


class BatcherConfig(BaseModel):
    # default value is suitable for 7B model in A100 GPU, this controls prefill rate of each step
    batch_max_tokens: int = Field(default=56000)
    # default value is suitable for 7B model in A100 GPU, this controls batch size of each step
    batch_max_beams: int = Field(default=32)


class CacheConfig(BaseModel):
    num_blocks: Optional[int] = Field(default=2500)  # default value is suitable for 7B model in A100 GPU
    num_blocks_cpu: Optional[int] = Field(default=1024)
    block_size: int = Field(default=16)
    gpu_memory_utilization: float = Field(default=0.98)
    watermark: float = 0.01
    pin_memory_on_cpu: bool = Field(default=True)


class ModelLoadingConfig(BaseModel):
    model_type: str = Field(default=..., regex="(" + "|".join(list(MODEL_AUTO_TABLE.keys())) + ")", example="llama")
    model_name_or_path: str = Field(default=..., example="path_to_llama_model_dir")
    torch_dtype: str = Field(default="float16", regex="(" + "|".join(list(TORCH_FLOAT_DTYPE_MAP.keys())) + ")")
    tokenizer_name_or_path: Optional[str] = Field(default=None)
    use_fast_tokenizer: bool = Field(default=False)
    trust_remote_code: bool = Field(default=False)
    quantize_method: Optional[str] = Field(default=None, regex="(gptq|)")
    model_max_length: int = Field(default=2048)
    device: int = Field(default=0)
    gptq_model_base_name: Optional[str] = Field(default=None)
    gptq_config_base_name: Optional[str] = Field(default=None)


class ParallelConfig(BaseModel):
    tp_size: int = Field(default=1)


class ModelConfig:
    def __init__(self, model_config: PretrainedConfig, parallel_config: ParallelConfig):
        self.model_config = model_config
        self.parallel_config = parallel_config

    def get_hidden_size(self):
        return self.model_config.hidden_size

    def get_head_size(self):
        return self.model_config.hidden_size // self.model_config.num_attention_heads

    def get_num_heads(self):
        # For GPTBigCode:
        if getattr(self.model_config, "multi_query", False):
            # Multi-query attention, only one KV head.
            return 1
        # For Falcon:
        if getattr(self.model_config, "n_head_kv", None) is not None:
            return self.model_config.n_head_kv

        return self.model_config.num_attention_heads // self.parallel_config.tp_size

    def get_num_layers(self):
        return self.model_config.num_hidden_layers


__all__ = [
    "ModelFactory",
    "TORCH_FLOAT_DTYPE_MAP",
    "MODEL_AUTO_TABLE",
    "BatcherConfig",
    "CacheConfig",
    "ModelLoadingConfig",
    "ParallelConfig",
    "ModelConfig"
]
