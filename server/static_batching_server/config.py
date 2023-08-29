from typing import Dict, Optional, Union

from pydantic import BaseModel, Field


class BatcherConfig(BaseModel):
    package_max_workload: int = Field(default=1)
    packaging_interval_seconds: int = Field(default=2)


class WorkerConfig(BaseModel):
    model_id: str = Field(default=..., example="model_id")
    model_name_or_path: str = Field(default=..., example="model_name_or_path")
    tokenizer_name_or_path: Optional[str] = Field(default=None)
    revision: str = Field(default="main")
    low_cpu_mem_usage: bool = Field(default=True)
    torch_dtype: Union[str] = Field(default="float16", regex="(float16|bfloat16)")
    device: Optional[Union[int, str]] = Field(default=None)
    max_memory: Optional[Dict[Union[str, int], str]] = Field(default=None)
    device_map: Union[str, Dict[str, Union[int, str]]] = Field(default="auto")
    use_fast_tokenizer: bool = Field(default=False)
    trust_remote_code: bool = Field(default=False)
    use_safetensors: bool = Field(default=False)
    batch_size: int = Field(default=-1)  # -1 means execute all inputs together no matter how many they are
    is_gptq_quantized: bool = Field(default=False)


__all__ = [
    "BatcherConfig",
    "WorkerConfig"
]
