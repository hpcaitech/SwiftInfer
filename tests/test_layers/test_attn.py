import copy
from collections import OrderedDict
from types import MethodType

import pytest
import tensorrt as trt
import tensorrt_llm
import torch
from streaming_llm.pos_shift.modify_llama import llama_pos_shift_attention_forward
from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.builder import Builder
from tensorrt_llm.functional import Tensor
from tensorrt_llm.layers.attention import Attention
from tensorrt_llm.network import net_guard
from torch.testing import assert_close
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, _expand_mask

from swiftinfer.layers.llama_attention import (
    AttentionMaskType,
    KeyValueCacheParams,
    LlamaPosShiftAttention,
    PositionEmbeddingType,
)


def copy_from_hf_layer(trt_layer: Attention, hf_layer: LlamaAttention) -> None:
    with torch.no_grad():
        qkv_weight = torch.cat([hf_layer.q_proj.weight, hf_layer.k_proj.weight, hf_layer.v_proj.weight], dim=0)
        trt_layer.qkv.weight.value = torch_to_numpy(qkv_weight.detach().cpu())
        trt_layer.dense.weight.value = torch_to_numpy(hf_layer.o_proj.weight.detach().cpu())


def run_trt_context(hidden_states, attention_mask, past_key_value,  hidden_size: int, num_attention_heads: int, max_seq_len: int, input_len: int, init_kvcache_size: int, hf_layer: LlamaAttention, use_custom_attn: bool = True, is_context_stage: bool=True):
    builder = Builder()
    builder_config = builder.create_builder_config(precision="float32")
    engine_name = "attn_layer"

    layer_cls = LlamaPosShiftAttention if use_custom_attn else Attention

    layer = layer_cls(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_mask_type=AttentionMaskType.causal,
        bias=False,
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
        dtype=trt.float32,
    )
    copy_from_hf_layer(layer, hf_layer)

    network = builder.create_network()
    network.trt_network.name = engine_name

    hidden_states_len = [0, input_len, hidden_size] if is_context_stage else [1, 1, 1]
    attention_mask_len = [0, input_len + init_kvcache_size, max_seq_len] if is_context_stage else [0, input_len + init_kvcache_size + 1, max_seq_len]
    kv_cache_len = [0, init_kvcache_size, max_seq_len] if is_context_stage else [0, input_len + init_kvcache_size, max_seq_len]

    with net_guard(network):
        network.set_named_parameters(layer.named_parameters())
        hidden_states_ = Tensor(
            "hidden_states",
            trt.float32,
            shape=[-1, -1, hidden_size],
            dim_range=OrderedDict(
                [
                    ("batch_size_fake", [[1, 1, 1]]),
                    ("input_len", [hidden_states_len]),
                    ("hidden_size", [hidden_size]),
                ]
            ),
        )
        attention_mask_ = Tensor(
            "attention_mask",
            trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict(
                [
                    ("batch_size_fake", [[1, 1, 1]]),
                    ("num_tokens", [attention_mask_len]),
                ]
            ),
        )
        kv_ = Tensor(
            "past_key_value_0",
            trt.float32,
            shape=[-1, 2, num_attention_heads, -1, hidden_size // num_attention_heads],
            dim_range=OrderedDict(
                [
                    ("batch_size_fake", [[1, 1, 1]]),
                    ("num_components", [2]),
                    ("num_attention_heads", [num_attention_heads]),
                    ("num_kv_cache", [kv_cache_len]),
                    ("head_size", [hidden_size // num_attention_heads]),
                ]
            ),
        )
 
        output, present_kv = layer(hidden_states_, attention_mask_, use_cache=True, kv_cache_params=KeyValueCacheParams(past_key_value=[kv_]))
        output.mark_output("output", output.dtype)
        present_kv.mark_output("present_kv", present_kv.dtype)

    tensorrt_llm.graph_rewriting.optimize(network)
    serialized_engine = builder.build_engine(network, builder_config)
    logger = trt.Logger()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    if is_context_stage:
        output = torch.zeros(1, input_len, hidden_size, device="cuda", dtype=torch.float32)
        present_kv = torch.zeros(1, 2, num_attention_heads, input_len + init_kvcache_size, hidden_size // num_attention_heads, device="cuda", dtype=torch.float32)

        if past_key_value is None:
            past_key_value_shape = (1, 2, num_attention_heads, 0, hidden_size // num_attention_heads)
            past_key_value = torch.zeros(1, device="cuda", dtype=torch.float32)
        else:
            past_key_value_shape = tuple(past_key_value.shape)
    else:
        output = torch.zeros(1, 1, hidden_size, device="cuda", dtype=torch.float32)
        present_kv = torch.zeros(1, 2, num_attention_heads, input_len + init_kvcache_size + 1, hidden_size // num_attention_heads, device="cuda", dtype=torch.float32)
        past_key_value_shape = tuple(past_key_value.shape)


    assert context.set_input_shape("hidden_states", tuple(hidden_states.shape))
    assert context.set_input_shape("attention_mask", tuple(attention_mask.shape))
    assert context.set_input_shape("past_key_value_0", past_key_value_shape)
    context.set_tensor_address("hidden_states", hidden_states.data_ptr())
    context.set_tensor_address("attention_mask", attention_mask.data_ptr())
    context.set_tensor_address("past_key_value_0", past_key_value.data_ptr())
    context.set_tensor_address("output", output.data_ptr())
    context.set_tensor_address("present_kv", present_kv.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return output, present_kv


def run_streamllm_context(hidden_states, attention_mask, past_key_value, hidden_size: int, num_attention_heads: int, max_seq_len: int, input_len: int, init_kvcache_size: int, layer: LlamaAttention, use_streaming_llm: bool=True, is_context_stage: bool=True):
    layer = layer.cuda()
    if use_streaming_llm:
        layer.forward = MethodType(llama_pos_shift_attention_forward, layer)
    layer.eval()
    if is_context_stage:
        position_ids = torch.arange(init_kvcache_size, input_len+init_kvcache_size, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    else:
        position_ids = torch.tensor([input_len + init_kvcache_size], dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    attention_mask = _expand_mask(attention_mask, hidden_states.dtype, hidden_states.size(1))
    if past_key_value is not None:
        past_key_value = past_key_value.unbind(dim=1)
    with torch.no_grad():
        output, _, present_key_value = layer(
            hidden_states,
            attention_mask,
            position_ids,
            use_cache=True,
            past_key_value=past_key_value,
        )
    present_key_value = torch.stack(present_key_value, dim=1)
    return output, present_key_value

@pytest.mark.parametrize("init_kvcache_size", [0, 4])
@pytest.mark.parametrize("is_context_stage", [True, False])
def test_attn(init_kvcache_size: int, is_context_stage: bool):
    torch.cuda.set_device(0)
    
    # test non-first round of inference
    # I is initial kv cache size
    # T is generation step number, start from 1
    # context stage
    # hidden_state: [B, S, H]
    # attn_mask: [B, S+I]
    # past_kv_cache: [B, 2, N, I, D]
    # present_kv_cache: [B, 2, N, S+I, D]
    # decode stage
    # hidden_state: [B, 1, H]
    # attn_mask: [B, S+I+T]
    # past_kv_cache: [B, 2, N, S+I+T-1, D]
    # present_kv_cache: [B, 2, N, S+I+T, D]

    hidden_size = 128
    num_attention_heads = 32
    max_seq_len = 256
    input_len = 64
    if is_context_stage:
        hidden_states = torch.rand(1, input_len, hidden_size, device="cuda", dtype=torch.float32)
        attention_mask = torch.ones(1, input_len + init_kvcache_size, device="cuda", dtype=torch.int64)
        if init_kvcache_size > 0:
            past_key_value = torch.rand(1, 2, num_attention_heads, init_kvcache_size, hidden_size // num_attention_heads, device="cuda", dtype=torch.float32)
        else:
            past_key_value = None
    else:
        hidden_states = torch.rand(1, 1, hidden_size, device="cuda", dtype=torch.float32)
        attention_mask = torch.ones(1, input_len + init_kvcache_size + 1, device="cuda", dtype=torch.int64)
        past_key_value = torch.rand(1, 2, num_attention_heads, input_len + init_kvcache_size, hidden_size // num_attention_heads, device="cuda", dtype=torch.float32)

    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_seq_len,
    )
    layer = LlamaAttention(config).cuda()
    clonded_layer = copy.deepcopy(layer) 
    output, kv_cache = run_streamllm_context(hidden_states, attention_mask, past_key_value, hidden_size, num_attention_heads, max_seq_len, input_len, init_kvcache_size, clonded_layer, use_streaming_llm=True, is_context_stage=is_context_stage)
    trt_output, trt_kv_cache = run_trt_context(hidden_states, attention_mask, past_key_value, hidden_size, num_attention_heads, max_seq_len, input_len, init_kvcache_size, layer, use_custom_attn=True, is_context_stage=is_context_stage)
    if is_context_stage:
        atol = 0.8 if init_kvcache_size > 0 else 0.5
    else:
        atol = 0.08
    assert_close(trt_output, output, rtol=0, atol=atol)
    assert_close(trt_kv_cache, kv_cache, rtol=0, atol=1e-3)



if __name__ == "__main__":
    test_attn(0, True)
    test_attn(4, True)
    test_attn(0, False)
    test_attn(4, False)