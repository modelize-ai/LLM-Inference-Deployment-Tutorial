from glob import glob
from os import path
from typing import *

import torch
from safetensors import safe_open


class Weights:
    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        dtype: torch.dtype,
        quantize_method: Optional[str] = None,
        gptq_model_base_name: Optional[str] = None
    ):
        if not path.isdir(model_name_or_path):
            raise NotADirectoryError(f"{model_name_or_path} not exists.")
        routing = {}
        file_pattern = "*.safetensors"
        if quantize_method == "gptq":
            file_pattern = "gptq_model*.safetensors"
            if gptq_model_base_name:
                file_pattern = f"{gptq_model_base_name}*.safetensors"
        for model_file in glob(path.join(model_name_or_path, file_pattern)):
            with safe_open(model_file, framework="pt") as f:
                for k in f.keys():
                    if k in routing:
                        raise RuntimeError(
                            f"Key {k} was found in multiple files: {model_file} and {routing[k]}"
                        )
                    routing[k] = model_file
        self.routing = routing
        self.device = device
        self.dtype = dtype
        self._handles = {}

    def _get_handle(self, filename: str):
        if filename not in self._handles:
            f = safe_open(filename, framework="pt")
            self._handles[filename] = f

        return self._handles[filename]

    def get_filename(self, tensor_name: str) -> (str, str):
        filename = self.routing.get(tensor_name, None)
        if filename is None:
            raise RuntimeError(f"weight {tensor_name} does not exist")
        return str(filename), tensor_name

    def _get_slice(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        return slice_

    def get_shape(self, tensor_name: str):
        return self._get_slice(tensor_name).get_shape()

    def get_tensor(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        tensor = f.get_tensor(tensor_name)
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype not in [torch.int32, torch.int64]:
            tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_gptq_weight(self, prefix: str):
        try:
            qweight = self.get_tensor(f"{prefix}.qweight")
        except RuntimeError:
            raise RuntimeError(
                "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
            )
        qzeros = self.get_tensor(f"{prefix}.qzeros")
        scales = self.get_tensor(f"{prefix}.scales")
        g_idx = self.get_tensor(f"{prefix}.g_idx")
        try:
            bias = self.get_tensor(f"{prefix}.bias")
        except:
            bias = None

        return qweight, qzeros, scales, g_idx, bias
