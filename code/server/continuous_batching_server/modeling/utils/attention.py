from typing import Optional, Tuple

import torch
import torch.nn as nn
import xformers.ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask, LowerTriangularMask

import vllm.attention_ops as vllm_attention_ops
import vllm.cache_ops as vllm_cache_ops
import vllm.pos_encoding_ops as vllm_pos_encoding_ops


def prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_ops: Optional[xops.AttentionOp] = None,
    attention_bias: Optional[xops.AttentionBias] = None,
    dropout: float = 0.0,
    scale: Optional[float] = None
):
    for each in [query, key, value]:
        if len(each.shape) != 4:
            raise ValueError(
                "input tensor must have 4-dim shape which are [bsz, seq_len, num_heads, head_size] respectively,"
                f"but get {each.shape}"
            )

    bsz, seq_len = query.shape[:2]

    if value.shape[2] != query.shape[2]:
        # MQA expand
        if value.shape[2] == 1:
            pass  # TODO
        # GQA reshape
        else:
            original_shape = value.shape
            pass  # TODO

    return xops.memory_efficient_attention(
        query=query,
        key=key,
        value=value,
        attn_bias=attention_bias,
        p=dropout,
        scale=scale,
        op=attention_ops
    ).reshape(bsz, seq_len, -1)


def decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_head_mapping: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor] = None
):
    if len(query.shape) != 3:
        raise ValueError(
            "query must have 3-dim shape which are [seq_len, num_heads, head_size] respectively, "
            f"but get shape {query.shape}"
        )

    attention_output = torch.empty_like(query)
    block_size = value_cache.shape[-1]
    vllm_attention_ops.single_query_cached_kv_attention(
        attention_output,
        query,
        key_cache,
        value_cache,
        kv_head_mapping,
        scale,
        block_tables,
        context_lengths,
        block_size,
        context_lengths.max().item(),
        alibi_slopes
    )
    return attention_output


class AttentionWithRoPE(nn.Module):
    def __init__(
        self,
        qkv_proj: nn.Module,
        out_proj: nn.Module,
        cos_sin_cache: torch.Tensor,
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        attention_ops: Optional[xops.AttentionOp] = None
    ):
        super(AttentionWithRoPE, self).__init__()

        self.qkv_proj = qkv_proj
        self.out_proj = out_proj

        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

        self.num_query_heads = num_query_heads
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads

        self.dropout = dropout
        self.scale = scale

        self.attention_ops = attention_ops

        # TODO: for now only compatible with GQA, make it also compatible with MQA
        self.num_groups = self.num_query_heads // self.num_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_value_heads, dtype=torch.int32, device=cos_sin_cache.device
        ).repeat_interleave(self.num_groups)

    def _qkv_forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_size = hidden_states.shape[-1]
        # for each out tensor, shape ==> [total_tokens, hidden_size]
        return self.qkv_proj(hidden_states.view(-1, hidden_size)).chunk(chunks=3, dim=-1)

    def _rope_forward(self, query: torch.Tensor, key: torch.Tensor, position_ids: Optional[torch.Tensor]) -> None:
        if position_ids is None:
            return

        hidden_size = query.shape[-1]
        position_ids = position_ids.view(-1)

        vllm_pos_encoding_ops.rotary_embedding_neox(
            position_ids,
            query,
            key,
            hidden_size // self.num_query_heads,
            self.cos_sin_cache
        )

    def _out_forward(self, hidden_states: torch.Tensor, shape: tuple) -> torch.Tensor:
        return self.out_proj(hidden_states).view(shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        prefill: bool,
        block_tables: Optional[torch.Tensor],
        slots: Optional[torch.Tensor],
        context_lengths: torch.Tensor,
        cache_event: Optional[torch.cuda.Event] = None
    ) -> torch.Tensor:
        # The shape of hidden_states and position_ids:
        # if is prefill ==> [bsz, max(context_lengths), hidden_size]
        # otherwise     ==> [bsz, 1, hidden_size]
        if len(hidden_states.shape) != 3:
            raise ValueError("hidden_states must have 3-dim shape.")
        bsz, max_len, hidden_size = hidden_states.shape

        # QKV projection
        query, key, value = self._qkv_forward(hidden_states)  # for each: shape ==> [total_tokens, hidden_size]

        # Add RoPE info
        self._rope_forward(query, key, position_ids)

        # Prefill Attention
        if prefill:
            attn_out = prefill_attention(
                query.view(bsz, max_len, self.num_query_heads, -1),
                key.view(bsz, max_len, self.num_key_heads, -1),
                value.view(bsz, max_len, self.num_value_heads, -1),
                self.attention_ops,
                LowerTriangularMask(),
                self.dropout,
                self.scale
            )

        # Wait until the cache op is done
        if cache_event is not None:
            cache_event.wait()

        # Cache key and value
        if kv_cache is not None:
            if prefill:
                valid_token_indices = []
                for i, start_idx in enumerate(range(0, bsz * max_len, max_len)):
                    end_idx = start_idx + max_len
                    indices = list(range(start_idx, end_idx))[-context_lengths[i]:]
                    valid_token_indices += indices
                key_to_cache = key[valid_token_indices]
                value_to_cache = value[valid_token_indices]
            else:
                key_to_cache = key[:len(slots)]
                value_to_cache = value[:len(slots)]
            num_valid_tokens = key_to_cache.shape[0]
            key_to_cache = key_to_cache.reshape(num_valid_tokens, self.num_key_heads, -1)
            value_to_cache = value_to_cache.reshape(num_valid_tokens, self.num_value_heads, -1)
            vllm_cache_ops.reshape_and_cache(
                key_to_cache, value_to_cache, kv_cache[0], kv_cache[1], slots
            )
        elif not prefill:
            raise ValueError("kv_cache can't be None when in decode stage.")

        # Decode Attention
        if not prefill:
            attn_out = decode_attention(
                query.view(bsz * max_len, self.num_query_heads, -1),
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.scale,
                block_tables,
                context_lengths,
                None
            ).view(bsz, max_len, -1)
        return self._out_forward(attn_out, (bsz, max_len, hidden_size))

    @staticmethod
    def build_rope_cache(
        rotary_dim: int,
        max_position: int = 2048,
        base: int = 10000,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.float16
    ):
        inv_freq = (1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim)))
        t = torch.arange(max_position, device=device, dtype=dtype)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        return cache


class VarLenAttentionWithRoPE(AttentionWithRoPE):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        prefill: bool,
        block_tables: Optional[torch.Tensor],
        slots: Optional[torch.Tensor],
        context_lengths: torch.Tensor,
        cache_event: Optional[torch.cuda.Event] = None
    ) -> torch.Tensor:
        # The shape of hidden_states and position_ids for both prefill and decode:
        # [total_tokens, hidden_size]

        # QKV projection
        query, key, value = self._qkv_forward(hidden_states)  # for each: shape ==> [total_tokens, hidden_size]

        # Add RoPE info
        self._rope_forward(query, key, position_ids)

        total_tokens = query.shape[0]
        query = query.view(total_tokens, self.num_query_heads, -1)
        key = key.view(total_tokens, self.num_key_heads, -1)
        value = value.view(total_tokens, self.num_value_heads, -1)

        # Prefill Attention
        if prefill:
            attn_out = prefill_attention(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                self.attention_ops,
                BlockDiagonalCausalMask.from_seqlens(context_lengths.tolist()),
                self.dropout,
                self.scale
            ).squeeze(0)

        # Wait until the cache op is done
        if cache_event is not None:
            cache_event.wait()

        # Cache key and value
        if kv_cache is not None:
            vllm_cache_ops.reshape_and_cache(
                key[:len(slots)], value[:len(slots)], kv_cache[0], kv_cache[1], slots
            )
        elif not prefill:
            raise ValueError("kv_cache can't be None when in decode stage.")

        # Decode Attention
        if not prefill:
            attn_out = decode_attention(
                query,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.scale,
                block_tables,
                context_lengths,
                None
            ).view(total_tokens, -1)

        return self._out_forward(attn_out, hidden_states.shape)


__all__ = ["AttentionWithRoPE", "VarLenAttentionWithRoPE"]
