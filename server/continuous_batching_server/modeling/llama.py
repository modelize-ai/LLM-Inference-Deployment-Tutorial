# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import *

import torch
import torch.nn as nn
import xformers.ops as xops
from accelerate import init_empty_weights
from auto_gptq import BaseQuantizeConfig as GPTQConfig
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig


from .utils.attention import VarLenAttentionWithRoPE
from .utils.linear import DynamicLinear
from .utils.weights import Weights


class LlamaConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    @classmethod
    def load(cls, prefix: str, weights: Weights, eps: float = 1e-6):
        weight = weights.get_tensor(f"{prefix}.weight")
        with init_empty_weights():
            ln = cls(weight.shape[0], eps)
        ln.weight = nn.Parameter(weight)
        return ln


def _load_gqa(config, prefix: str, weights: Weights, gptq_config: Optional[GPTQConfig] = None):
    w = [
        weights.get_tensor(f"{prefix}.q_proj.weight"),
        weights.get_tensor(f"{prefix}.k_proj.weight"),
        weights.get_tensor(f"{prefix}.v_proj.weight")
    ]
    weight = torch.cat(w, dim=0)
    weight = weight.to(dtype=weights.dtype).to(device=weights.device)

    assert config.hidden_size % config.num_attention_heads == 0
    head_size = config.hidden_size // config.num_attention_heads
    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    assert list(weight.shape) == [
        (num_heads + 2 * num_key_value_heads) * head_size,
        config.hidden_size,
    ], f"{list(weight.shape)} != {[(num_heads + 2 * config.num_key_value_heads) * head_size, config.hidden_size]}"

    return DynamicLinear.load(config, prefix, weights, False, gptq_config)


class LlamaMLP(nn.Module):
    def __init__(self, prefix: str, config: LlamaConfig, weights: Weights, gptq_config: Optional[GPTQConfig] = None):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )
        # Fuse gate and up proj
        self.gate_up_proj = DynamicLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
            gptq_config=gptq_config
        )
        self.down_proj = DynamicLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
            gptq_config=gptq_config
        )
        self.intermediate_size = config.intermediate_size

    def forward(self, hidden_states):
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class LlamaLayer(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig, weights: Weights, gptq_config: Optional[GPTQConfig] = None):
        super().__init__()

        prefix = f"model.layers.{layer_id}"

        self.config = config
        self.gptq_config = gptq_config

        self.self_attn = self._init_attention_module(config, f"{prefix}.self_attn", weights)
        self.mlp = LlamaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights, gptq_config=gptq_config)

        self.input_layernorm = LlamaRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=config.rms_norm_eps
        )

    def _init_attention_module(self, config: LlamaConfig, prefix: str, weights: Weights) -> nn.Module:
        qkv_proj = DynamicLinear.load_multi(
            config=config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            weights=weights,
            bias=False,
            dim=0,
            gptq_config=self.gptq_config
        )
        o_proj = DynamicLinear.load(
            config=config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
            gptq_config=self.gptq_config
        )

        cos_sin_cache = VarLenAttentionWithRoPE.build_rope_cache(
            rotary_dim=config.hidden_size // config.num_attention_heads,
            max_position=config.max_position_embeddings,
            base=10000,
            device=o_proj.weight.device if not self.gptq_config else o_proj.scales.device,
            dtype=o_proj.weight.dtype if not self.gptq_config else o_proj.scales.dtype
        )

        head_dim = config.hidden_size // config.num_attention_heads
        attn_fw_op = xops.fmha.flash.FwOp if head_dim <= 128 else xops.fmha.cutlass.FwOp
        return VarLenAttentionWithRoPE(
            qkv_proj=qkv_proj,
            out_proj=o_proj,
            cos_sin_cache=cos_sin_cache,
            num_query_heads=config.num_attention_heads,
            num_key_heads=config.num_key_value_heads,
            num_value_heads=config.num_key_value_heads,
            dropout=0.0,
            scale=head_dim ** -0.5,
            attention_ops=(attn_fw_op, None)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefill: bool,
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        context_lengths: torch.Tensor,
        cache_event: Optional[torch.cuda.Event] = None,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            position_ids,
            kv_cache,
            prefill,
            block_tables,
            slots,
            context_lengths,
            cache_event
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(torch.nn.Module):
    def __init__(self, config: LlamaConfig, weights: Weights, gptq_config: Optional[GPTQConfig] = None):
        super().__init__()
        self.config = config
        self.gptq_config = gptq_config

        self.embed_tokens = nn.Embedding.from_pretrained(
            embeddings=weights.get_tensor("model.embed_tokens.weight"),
            freeze=True,
            padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                LlamaLayer(
                    layer_id,
                    config,
                    weights,
                    gptq_config
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm.load(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefill: bool,
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        context_lengths: torch.Tensor,
        cache_events: Optional[List[torch.cuda.Event]] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_ids,
                kv_cache[i],
                prefill,
                block_tables,
                slots,
                context_lengths,
                cache_events[i] if cache_events is not None else None
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(torch.nn.Module):
    def __init__(self, config: LlamaConfig, weights: Weights, gptq_config: Optional[GPTQConfig] = None):
        super().__init__()

        self.config = config
        self.gptq_config = gptq_config
        self.model = LlamaModel(config, weights, gptq_config)
        self.lm_head = DynamicLinear.load(
            config,
            prefix="lm_head",
            weights=weights,
            bias=False,
            gptq_config=None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        prefill: bool,
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        context_lengths: torch.Tensor,
        cache_events: Optional[List[torch.cuda.Event]] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            position_ids,
            kv_cache,
            prefill,
            block_tables,
            slots,
            context_lengths,
            cache_events,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)
        return logits
