from typing import *

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from auto_gptq import BaseQuantizeConfig as GPTQConfig
from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
from torch.nn import functional as F

from .weights import Weights


class FastLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    @classmethod
    def load(
        cls,
        config,
        prefix: str,
        weights: Weights,
        bias: bool
    ):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    @classmethod
    def load_multi(
        cls,
        config,
        prefixes: List[str],
        weights: Weights,
        bias: bool,
        dim: int
    ):
        w = [
            weights.get_tensor(f"{prefix}.weight") for prefix in prefixes
        ]
        weight = torch.cat(w, dim=dim)

        if bias:
            b = [weights.get_tensor(f"{p}.bias") for p in prefixes]
            bias = torch.cat(b, dim=dim)
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class DynamicLinear:
    @classmethod
    def load(
        cls,
        config,
        prefix: str,
        weights: Weights,
        bias: bool,
        gptq_config: Optional[GPTQConfig] = None
    ):
        if not gptq_config:
            return FastLinear.load(config, prefix, weights, bias)

        disable_exllama = False
        if gptq_config.bits != 4 or gptq_config.desc_act:  # for the later condition, can be removed once auto-gptq fixed it
            disable_exllama = True
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False,
            desc_act=gptq_config.desc_act,
            group_size=gptq_config.group_size,
            bits=gptq_config.bits,
            disable_exllama=disable_exllama
        )

        qweight, qzeros, scales, g_idx, bias = weights.get_gptq_weight(prefix)

        init_args = (
            gptq_config.bits,
            gptq_config.group_size,
            qweight.shape[0] * 32 // gptq_config.bits,
            qweight.shape[1],
            bias is not None
        )
        with init_empty_weights(include_buffers=True):
            quant_linear = QuantLinear(*init_args, trainable=False)
        quant_linear.qweight = qweight
        quant_linear.qzeros = qzeros
        quant_linear.scales = scales
        quant_linear.g_idx = g_idx
        quant_linear.bias = bias

        return quant_linear

    @classmethod
    def load_multi(
        cls,
        config,
        prefixes: List[str],
        weights: Weights,
        bias: bool,
        dim: int,
        gptq_config: Optional[GPTQConfig] = None
    ):
        if not gptq_config:
            return FastLinear.load_multi(config, prefixes, weights, bias, dim)

        disable_exllama = False
        if gptq_config.bits != 4 or gptq_config.desc_act:  # for the later condition, can be removed once auto-gptq fixed it
            disable_exllama = True
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False,
            desc_act=gptq_config.desc_act,
            group_size=gptq_config.group_size,
            bits=gptq_config.bits,
            disable_exllama=disable_exllama
        )

        qweight_li, qzeros_li, scales_li, g_idx_li, bias_li = [], [], [], [], []
        outfeatures = 0
        for prefix in prefixes:
            qweight, qzeros, scales, g_idx, bias = weights.get_gptq_weight(prefix)
            qweight_li.append(qweight)
            qzeros_li.append(qzeros)
            scales_li.append(scales)
            g_idx_li.append(g_idx)
            bias_li.append(bias)
            outfeatures += qweight.shape[1]

        qweight = torch.cat(qweight_li, dim=1)
        qzeros = torch.cat(qzeros_li, dim=1)
        scales = torch.cat(scales_li, dim=1)
        g_idx = torch.cat(g_idx_li, dim=0)
        if bias_li[0] is not None:
            bias = torch.cat(bias_li, dim=0)
        else:
            bias = None

        init_args = (
            gptq_config.bits,
            gptq_config.group_size,
            qweight.shape[0] * 32 // gptq_config.bits,
            qweight.shape[1],
            bias is not None
        )

        with init_empty_weights(include_buffers=True):
            quant_linear = QuantLinear(*init_args, trainable=False)
        quant_linear.qweight = qweight
        quant_linear.qzeros = qzeros
        quant_linear.scales = scales
        quant_linear.g_idx = g_idx
        quant_linear.bias = bias

        return quant_linear
