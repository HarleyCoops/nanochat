"""
Hugging Face-compatible nanochat Transformer implementation.

This file mirrors the architecture used during training (RoPE, RMSNorm,
multi-query attention, relu^2 MLP, untied embeddings, logits softcap) while
presenting the familiar `PreTrainedModel` interface so that checkpoints can be
served directly from the Hugging Face Hub.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import AutoConfig, AutoModelForCausalLM

from configuration_nanochat import NanoChatConfig

logger = logging.get_logger(__name__)


def rms_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def relu_squared(x: Tensor) -> Tensor:
    return F.relu(x) ** 2


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return x
    b, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(b, n_kv_heads * n_rep, seq_len, head_dim)


class NanoChatAttention(nn.Module):
    def __init__(self, config: NanoChatConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        if config.n_embd % config.n_head != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, q_len, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.n_kv_head, self.head_dim).transpose(1, 2)

        query, key = apply_rotary_emb(query, key, cos, sin)
        if self.config.use_qk_norm:
            query = rms_norm(query)
            key = rms_norm(key)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            key = torch.cat([past_k, key], dim=2)
            value = torch.cat([past_v, value], dim=2)

        present = (key, value) if use_cache else None

        key_for_scores = repeat_kv(key, self.n_head // self.n_kv_head)
        value_for_scores = repeat_kv(value, self.n_head // self.n_kv_head)

        attn_scores = torch.matmul(query, key_for_scores.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.to(torch.float32)

        # causal mask that accounts for the prefix introduced by past key values
        if attn_scores.size(-1) != q_len:
            total_k = attn_scores.size(-1)
            past_len = total_k - q_len
            mask = torch.arange(total_k, device=attn_scores.device)
            causal = mask.unsqueeze(0) <= (mask.new_tensor(past_len) + torch.arange(q_len, device=mask.device).unsqueeze(1))
            attn_scores = attn_scores.masked_fill(~causal, torch.finfo(attn_scores.dtype).min)
        else:
            mask = torch.triu(torch.ones((q_len, q_len), device=attn_scores.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask, torch.finfo(attn_scores.dtype).min)

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, value_for_scores).to(value_for_scores.dtype)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, present


class NanoChatMLP(nn.Module):
    def __init__(self, config: NanoChatConfig):
        super().__init__()
        hidden_dim = config.n_embd * 4
        self.fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(relu_squared(self.fc(x)))


class NanoChatBlock(nn.Module):
    def __init__(self, config: NanoChatConfig):
        super().__init__()
        self.attn = NanoChatAttention(config)
        self.mlp = NanoChatMLP(config)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        residual = x
        attn_input = rms_norm(x)
        attn_output, present = self.attn(attn_input, cos, sin, past_key_value, use_cache)
        x = residual + attn_output
        mlp_input = rms_norm(x)
        x = x + self.mlp(mlp_input)
        return x, present


class NanoChatModel(nn.Module):
    def __init__(self, config: NanoChatConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.blocks = nn.ModuleList([NanoChatBlock(config) for _ in range(config.n_layer)])

        self.softcap = config.softcap
        self._rope_cache: Optional[Tuple[Tensor, Tensor]] = None
        self._rope_cache_length = 0

    def _build_rope_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        if self._rope_cache is not None and self._rope_cache_length >= seq_len and self._rope_cache[0].device == device:
            return self._rope_cache

        head_dim = self.config.n_embd // self.config.n_head
        theta = 10000.0 ** (-torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        position_ids = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", position_ids, theta)
        cos = torch.repeat_interleave(freqs.cos()[None, None, :, :], repeats=1, dim=0)
        sin = torch.repeat_interleave(freqs.sin()[None, None, :, :], repeats=1, dim=0)
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)

        self._rope_cache = (cos, sin)
        self._rope_cache_length = seq_len
        return cos, sin

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], ...]]]:
        del attention_mask  # attention masking is handled implicitly via causal masking
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype

        inputs_embeds = self.embed_tokens(input_ids)
        x = inputs_embeds

        past_key_values = past_key_values or tuple([None] * len(self.blocks))
        past_length = past_key_values[0][0].size(2) if past_key_values and past_key_values[0] is not None else 0

        cos_full, sin_full = self._build_rope_cache(seq_len + past_length, device, dtype)
        cos = cos_full[:, :, past_length:, :]
        sin = sin_full[:, :, past_length:, :]
        new_past_key_values = [] if use_cache else None

        for layer, block in enumerate(self.blocks):
            past = past_key_values[layer] if past_key_values[layer] is not None else None
            x, present = block(x, cos, sin, past, use_cache)
            if use_cache:
                new_past_key_values.append(present)

        x = rms_norm(x)
        logits = self.lm_head(x)

        if self.softcap is not None and self.softcap > 0:
            logits = self.softcap * torch.tanh(logits / self.softcap)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)

        return logits, loss, tuple(new_past_key_values) if use_cache else None


class NanoChatForCausalLM(PreTrainedModel):
    config_class = NanoChatConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def __init__(self, config: NanoChatConfig):
        super().__init__(config)
        self.model = NanoChatModel(config)
        if config.tie_word_embeddings:
            self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.model.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": kwargs.get("use_cache", True)}

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered = []
        for layer_past in past_key_values:
            reordered.append(
                (
                    layer_past[0].index_select(0, beam_idx),
                    layer_past[1].index_select(0, beam_idx),
                )
            )
        return tuple(reordered)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        logits, loss, new_past = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
        )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past,
        )


AutoConfig.register("nanochat", NanoChatConfig)
AutoModelForCausalLM.register(NanoChatConfig, NanoChatForCausalLM)
