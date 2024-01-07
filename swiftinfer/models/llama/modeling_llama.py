# Copyright 2023 NVIDIA Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is modified from
# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.6.0/tensorrt_llm/models/llama/model.py

from typing import List, Optional

import tensorrt as trt
from tensorrt_llm._common import default_net
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_trt
from tensorrt_llm.functional import Tensor, gather_last_token_logits, recv, send
from tensorrt_llm.layers import (
    AttentionMaskType,
    AttentionParams,
    ColumnLinear,
    Embedding,
    FusedGatedMLP,
    GatedMLP,
    KeyValueCacheParams,
    LoraParams,
    PromptTuningEmbedding,
    RmsNorm,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.quantization import QuantMode

from swiftinfer.layers.llama_attention import AttentionParams, LlamaPosShiftAttention, PositionEmbeddingType

from ..generation_utils import GenerationMixin


class LLaMADecoderLayer(Module):
    def __init__(
        self,
        layer_id,
        hidden_size,
        num_attention_heads,
        num_kv_heads=None,
        max_position_embeddings=2048,
        dtype=None,
        attention_mask_type=AttentionMaskType.causal,
        hidden_act="silu",
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
        rotary_base=10000.0,
        rotary_scaling=None,
        mlp_hidden_size=None,
        tp_group=None,
        tp_size=1,
        quant_mode=QuantMode(0),
        rms_norm_eps=1e-06,
        attn_bias=False,
        mlp_bias=False,
        use_fused_mlp=False,
    ):
        super().__init__()
        self._layer_id = layer_id  # useful for debugging
        # used for quantizing model
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.dtype = dtype
        self.hidden_act = hidden_act
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.mlp_hidden_size = mlp_hidden_size
        self.attention_mask_type = attention_mask_type
        self.position_embedding_type = position_embedding_type
        self.input_layernorm = RmsNorm(normalized_shape=hidden_size, eps=rms_norm_eps, dtype=dtype)

        self.attention = LlamaPosShiftAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=attn_bias,
            position_embedding_type=position_embedding_type,
            rotary_embedding_base=rotary_base,
            rotary_embedding_scaling=rotary_scaling,
            tp_group=tp_group,
            tp_size=tp_size,
            use_int8_kv_cache=quant_mode.has_int8_kv_cache(),
            quant_mode=quant_mode,
            instance_id=2 * layer_id,
        )
        if not mlp_hidden_size:
            self.mlp_hidden_size = hidden_size * 4
        ClsMLP = FusedGatedMLP if use_fused_mlp is True else GatedMLP
        self.mlp = ClsMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=self.mlp_hidden_size,
            hidden_act=hidden_act,
            dtype=dtype,
            bias=mlp_bias,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=quant_mode,
            instance_id=2 * layer_id + 1,
        )
        self.post_layernorm = RmsNorm(normalized_shape=hidden_size, eps=rms_norm_eps, dtype=dtype)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
        all_reduce_workspace=None,
        lora_param=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self._layer_id == 0:
            self.register_network_output(f"norm0", hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            workspace=all_reduce_workspace,
            lora_param=lora_param,
        )

        if use_cache:
            attention_output, presents = attention_output
        if self._layer_id == 0:
            self.register_network_output(f"attn", attention_output)

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        if self._layer_id == 0:
            self.register_network_output(f"norm1", hidden_states)

        hidden_states = self.mlp(hidden_states, all_reduce_workspace, lora_param=lora_param)
        if self._layer_id == 0:
            self.register_network_output(f"mlp", hidden_states)

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class LLaMAModel(Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        num_kv_heads,
        hidden_size,
        vocab_size,
        hidden_act,
        max_position_embeddings,
        dtype,
        mlp_hidden_size=None,
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
        rotary_base=10000.0,
        rotary_scaling=None,
        mapping=Mapping(),
        quant_mode=QuantMode(0),
        use_parallel_embedding=False,
        embedding_sharding_dim=0,
        rms_norm_eps=1e-06,
        use_fused_mlp=False,
        attn_bias=False,
        mlp_bias=False,
        use_prompt_tuning: bool = False,
    ):
        super().__init__()
        self.mapping = mapping
        self.use_prompt_tuning = use_prompt_tuning

        EmbeddingCls = PromptTuningEmbedding if use_prompt_tuning else Embedding
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = EmbeddingCls(
                num_embeddings=vocab_size,
                embedding_dim=hidden_size,
                dtype=dtype,
                tp_size=mapping.tp_size if use_parallel_embedding else 1,
                tp_group=mapping.tp_group if use_parallel_embedding else None,
                sharding_dim=embedding_sharding_dim,
                tp_rank=mapping.tp_rank,
                instance_id=2 * num_layers,  # ids in [0, 2 * (num_layers - 1) + 1] already used
            )

        self.layers = ModuleList(
            [
                LLaMADecoderLayer(
                    layer_id=i,
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    max_position_embeddings=max_position_embeddings,
                    dtype=dtype,
                    hidden_act=hidden_act,
                    mlp_hidden_size=mlp_hidden_size,
                    position_embedding_type=position_embedding_type,
                    rotary_base=rotary_base,
                    rotary_scaling=rotary_scaling,
                    tp_group=mapping.tp_group,
                    tp_size=mapping.tp_size,
                    quant_mode=quant_mode,
                    rms_norm_eps=rms_norm_eps,
                    attn_bias=attn_bias,
                    mlp_bias=mlp_bias,
                    use_fused_mlp=use_fused_mlp,
                )
                for i in self.get_transformer_layers(self.mapping, num_layers)
            ]
        )

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=hidden_size, eps=rms_norm_eps, dtype=dtype)

    def forward(
        self,
        input_ids,
        position_ids=None,
        use_cache=False,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        hidden_states=None,
        all_reduce_workspace=None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        lora_params=None,
    ):
        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        ptuning_args = []
        if self.use_prompt_tuning:
            ptuning_args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size]
        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args, all_reduce_workspace)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())
        self.register_network_output(f"embd", hidden_states)

        for layer_idx, (layer, past, pointer, host_pointer, max_kv_cache_length) in enumerate(
            zip(
                self.layers,
                kv_cache_params.past_key_value,
                kv_cache_params.kv_cache_block_pointers,
                kv_cache_params.host_kv_cache_block_pointers,
                kv_cache_params.host_max_kv_cache_lengths,
            )
        ):
            lora_param = None
            if lora_params.lora_ranks is not None:
                lora_param = lora_params.get_layer_params(layer_idx)

            hidden_states = layer(
                hidden_states,
                use_cache=use_cache,
                attention_mask=attention_mask,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.host_past_key_value_lengths,
                    host_max_kv_cache_lengths=max_kv_cache_length,
                    kv_cache_block_pointers=[pointer],
                    host_kv_cache_block_pointers=[host_pointer],
                    cache_indirection=kv_cache_params.cache_indirection,
                ),
                attention_params=attention_params,
                all_reduce_workspace=all_reduce_workspace,
                lora_param=lora_param,
            )

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class LLaMAForCausalLM(LLaMAModel, GenerationMixin):
    def __init__(
        self,
        num_layers,
        num_heads,
        num_kv_heads,
        hidden_size,
        vocab_size,
        hidden_act,
        max_position_embeddings,
        dtype,
        logits_dtype="float32",
        mlp_hidden_size=None,
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
        rotary_base=10000.0,
        rotary_scaling=None,
        mapping=Mapping(),
        quant_mode=QuantMode(0),
        use_parallel_embedding=False,
        embedding_sharding_dim=0,
        rms_norm_eps=1e-06,
        use_fused_mlp=False,
        attn_bias=False,
        mlp_bias=False,
        use_prompt_tuning: bool = False,
    ):
        if isinstance(dtype, str):
            self.dtype = str_dtype_to_trt(dtype)
        else:
            assert isinstance(dtype, trt.DataType)
            self.dtype = dtype

        if isinstance(logits_dtype, str):
            self.logits_dtype = str_dtype_to_trt(logits_dtype)
        else:
            assert isinstance(logits_dtype, trt.DataType)
            self.logits_dtype = logits_dtype

        self.num_layers = num_layers
        self.num_heads = num_heads
        if num_kv_heads is None or num_kv_heads <= 0:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_size = mapping.tp_size

        self.kv_dtype = self.dtype
        if quant_mode.has_int8_kv_cache():
            self.kv_dtype = str_dtype_to_trt("int8")
        elif quant_mode.has_fp8_kv_cache():
            self.kv_dtype = str_dtype_to_trt("fp8")

        self.quant_mode = quant_mode
        self.use_parallel_embedding = use_parallel_embedding
        self.embedding_sharding_dim = embedding_sharding_dim

        super().__init__(
            num_layers,
            num_heads,
            num_kv_heads,
            hidden_size,
            vocab_size,
            hidden_act,
            max_position_embeddings,
            dtype,
            mlp_hidden_size,
            position_embedding_type,
            rotary_base,
            rotary_scaling,
            mapping,
            quant_mode,
            use_parallel_embedding,
            embedding_sharding_dim,
            rms_norm_eps,
            use_fused_mlp,
            attn_bias,
            mlp_bias,
            use_prompt_tuning,
        )

        vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
        if self.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(
                hidden_size,
                vocab_size_padded,
                bias=False,
                dtype=dtype,
                tp_group=mapping.tp_group,
                tp_size=mapping.tp_size,
                gather_output=True,
            )

    def forward(
        self,
        input_ids,
        position_ids=None,
        use_cache=False,
        last_token_ids=None,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        hidden_states=None,
        all_reduce_workspace=None,
        prompt_embedding_table: Optional[Tensor] = None,
        prompt_tasks: Optional[Tensor] = None,
        prompt_vocab_size: Optional[Tensor] = None,
        lora_params=None,
    ):
        hidden_states = super().forward(
            input_ids,
            position_ids,
            use_cache,
            attention_mask,
            kv_cache_params,
            attention_params,
            hidden_states,
            all_reduce_workspace,
            prompt_embedding_table,
            prompt_tasks,
            prompt_vocab_size,
            lora_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids, default_net().plugin_config.remove_input_padding
            )

            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            lm_logits = self.lm_head(hidden_states)
            lm_logits.mark_output("logits", self.logits_dtype)
        else:
            hidden_states.mark_output("hidden_states_output", self.dtype)

        if use_cache and default_net().plugin_config.paged_kv_cache == False:
            for i, present in zip(self.get_transformer_layers(self.mapping, self.num_layers), presents):
                present.mark_output(f"present_key_value_{i}", self.kv_dtype)
            if self.mapping.is_last_pp_rank():
                return (lm_logits, presents)
            return (hidden_states, presents)
        else:
            if self.mapping.is_last_pp_rank():
                return lm_logits
            return hidden_states

    def prepare_inputs(
        self,
        max_batch_size,
        max_input_len,
        max_new_tokens,
        use_cache,
        max_beam_width,
        max_num_tokens: int = None,
        prompt_embedding_table_size: int = 0,
        gather_all_token_logits: bool = False,
        lora_target_modules: List[str] = None,
        max_kv_cache_size: int = 0,
    ):
        """@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the
        ranges of the dimensions of when using TRT dynamic shapes.

        @return: a list contains values which can be fed into the self.forward()
        """

        # Prepare inputs
        head_size = self.hidden_size // self.num_heads
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net().plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        use_custom_all_reduce = default_net().plugin_config.use_custom_all_reduce
        use_lora_plugin = default_net().plugin_config.lora_plugin

        model_inputs = self.prepare_basic_inputs(
            max_batch_size,
            max_beam_width,
            max_input_len,
            max_new_tokens,
            self.num_kv_heads,
            head_size,
            self.num_layers,
            self.kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            use_custom_all_reduce=use_custom_all_reduce,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            dtype=self.dtype,
            num_heads=self.num_heads,
            mapping=self.mapping,
            max_num_tokens=max_num_tokens,
            prompt_embedding_table_size=prompt_embedding_table_size,
            gather_all_token_logits=gather_all_token_logits,
            use_lora_plugin=use_lora_plugin,
            lora_target_modules=lora_target_modules,
            max_kv_cache_size=max_kv_cache_size,
        )

        return (
            model_inputs["input_ids"],
            model_inputs["position_ids"],
            True,
            model_inputs["last_token_ids"],
            model_inputs["attention_mask"],
            KeyValueCacheParams(
                past_key_value=model_inputs["past_key_value"],
                host_past_key_value_lengths=model_inputs["host_past_key_value_lengths"],
                host_max_kv_cache_lengths=model_inputs["host_max_kv_cache_lengths"],
                kv_cache_block_pointers=model_inputs["kv_cache_block_pointers_list"],
                host_kv_cache_block_pointers=model_inputs["host_kv_cache_block_pointers_list"],
                cache_indirection=model_inputs["cache_indirection"],
            ),
            AttentionParams(
                sequence_length=model_inputs["sequence_length"],
                context_lengths=model_inputs["context_lengths"],
                host_context_lengths=model_inputs["host_context_lengths"],
                max_context_length=max_input_len,
                host_request_types=model_inputs["host_request_types"],
            ),
            model_inputs["hidden_states_input"],
            model_inputs["all_reduce_workspace"],
            model_inputs["prompt_embedding_table"],
            model_inputs["tasks"],
            model_inputs["prompt_vocab_size"],
            LoraParams(
                model_inputs["lora_ranks"],
                model_inputs["lora_weights_pointers"],
                host_context_lengths=model_inputs["host_context_lengths"],
                max_context_length=max_input_len,
                host_request_types=model_inputs["host_request_types"],
            ),
        )
