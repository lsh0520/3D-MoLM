import math
import warnings
from typing import List, Optional, Tuple
import logging

import torch
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import rotate_half # apply_rotary_pos_emb

# from einops import rearrange
# from flash_attn.flash_attn_interface import (  # pip3 install "flash-attn>=2.0"
#     flash_attn_varlen_qkvpacked_func,
# )
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
)
from flash_attn.bert_padding import unpad_input, pad_input
from transformers.models.llama.modeling_llama import _make_causal_mask, _expand_mask, repeat_kv
# from functools import partialmethod
import torch.nn.functional as F


def apply_rotary_pos_emb(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    bsz = gather_indices.shape[0]
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )
    # shape: (b, s, num_heads, head_dim)

    kv_seq_len = k.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[1]
        kv_seq_len += past_kv_len

    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb(q, k, cos_sin, position_ids)

    if past_key_value is not None:
        # reuse k, v
        k = torch.cat([past_key_value[0], k], dim=1)
        v = torch.cat([past_key_value[1], v], dim=1)

    past_key_value = (k, v) if use_cache else None

    key_padding_mask = attention_mask
    # Ideally we could just do this:
    #  q, indices, cu_q_lens, max_s = unpad_input(q, key_padding_mask[:, -q_len:])
    # but this does not work as Flash attention treats the q seq and kv seq as starting at index 0
    # which then breaks the causality logic. Probably if q_len >> past_kv_len we should
    # just skip flash attention. Leaving this in for now to demonstrate correctness of
    # flash attention information even when q needs padding.
    # TODO(siddartha): delegate back to original implementation on this condition.
    if past_kv_len > 0:
        q = torch.cat(
            (
                torch.full(
                    (bsz, past_kv_len, self.num_heads, self.head_dim),
                    0.0,
                    dtype=q.dtype,
                    device=q.device,
                ),
                q,
            ),
            dim=1,
        )

    if key_padding_mask is None:
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len + past_kv_len, -1
        )
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, key_padding_mask)
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), key_padding_mask
        )
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len + past_kv_len)

    # Need to strip off the zero query outputs.
    if past_kv_len > 0:
        output = output[:, past_kv_len:, ...]

    return self.o_proj(output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as flash attention
# takes a boolean key_padding_mask. Fills in the past kv length for use in forward.
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask




# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _original_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def original_forward_v2(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    if not hasattr(self, "pretraining_tp"):
        self.pretraining_tp = 1
    
    if self.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# def original_forward(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_value: Optional[Tuple[torch.Tensor]] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#     bsz, q_len, _ = hidden_states.size()

#     query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

#     kv_seq_len = key_states.shape[-2]
#     if past_key_value is not None:
#         kv_seq_len += past_key_value[0].shape[-2]
#     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
#     # [bsz, nh, t, hd]

#     if past_key_value is not None:
#         # reuse k, v, self_attention
#         key_states = torch.cat([past_key_value[0], key_states], dim=2)
#         value_states = torch.cat([past_key_value[1], value_states], dim=2)

#     past_key_value = (key_states, value_states) if use_cache else None

#     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

#     if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#         raise ValueError(
#             f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
#             f" {attn_weights.size()}"
#         )

#     if attention_mask is not None:
#         if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#             raise ValueError(
#                 f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
#             )
#         attn_weights = attn_weights + attention_mask
#         attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

#     # upcast attention to fp32
#     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#     attn_output = torch.matmul(attn_weights, value_states)

#     if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#         raise ValueError(
#             f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#             f" {attn_output.size()}"
#         )

#     attn_output = attn_output.transpose(1, 2)
#     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#     attn_output = self.o_proj(attn_output)

#     if not output_attentions:
#         attn_weights = None

#     return attn_output, attn_weights, past_key_value


def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward

def replace_flash_attn_with_llama_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _original_prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = original_forward_v2



# def replace_flash_attn_with_original_attn():
#     cuda_major, cuda_minor = torch.cuda.get_device_capability()
#     if cuda_major < 8:
#         logging.warning(
#             "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
#             "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
#         )
#     transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _original_prepare_decoder_attention_mask
#     transformers.models.llama.modeling_llama.LlamaAttention.forward = original_forward_v2


def set_llama_mode(mode):
    if mode == 'flash_decoder':
        ## compatible with opt's forward functions and training, but does not support opt's generate function
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        if cuda_major < 8:
            logging.warning(
                "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
                "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            )
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    elif mode == 'original_decoder':
        ## support everything, but slower and consumes more memory
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _original_prepare_decoder_attention_mask
        transformers.models.llama.modeling_llama.LlamaAttention.forward = original_forward_v2
    else:
        raise NotImplementedError(f"mode {mode} not implemented")



