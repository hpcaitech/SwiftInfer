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
# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.6.0/tensorrt_llm/runtime/generation.py

import math
from typing import Dict, List, Optional

from tensorrt_llm._ipc_utils import IpcMemory, set_peer_access
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.generation import GenerationSession as _GenerationSession
from tensorrt_llm.runtime.generation import RuntimeTensor, SamplingConfig
from tensorrt_llm.runtime.lora_manager import LoraManager

# isort: off
import torch
import tensorrt as trt


class GenerationSession(_GenerationSession):
    def setup(
        self,
        batch_size: int,
        max_context_length: int,
        max_new_tokens: int,
        beam_width: int = 1,
        max_kv_cache_length: Optional[int] = None,
        encoder_max_input_length: Optional[int] = None,
        lora_manager: LoraManager = None,
        lora_uids: List[str] = None,
        past_key_values: Optional[Dict[int, torch.Tensor]] = None,
    ):
        # Store these params related to buffer size to check against
        # the input shape with the params given in decode()
        self.latest_buffer = None
        self.early_stopped = False
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_context_length + max_new_tokens
        self.beam_width = beam_width
        self.encoder_max_input_length = encoder_max_input_length
        self.use_streaming_llm = past_key_values is not None
        self.init_past_key_values = past_key_values
        if self.use_streaming_llm and self.use_gpt_attention_plugin:
            logger.warning(
                "Streaming LLM is not supported with GPT attention plugin. "
                "Please use the default attention plugin instead."
            )
        if self.use_streaming_llm:
            self.init_kv_cache_size = next(iter(past_key_values.values())).shape[3]
        else:
            self.init_kv_cache_size = 0
        if max_kv_cache_length is None:
            self.max_kv_cache_length = self.max_seq_length + self.init_kv_cache_size
            logger.debug("The max_kv_cache_length is not set, we will use max_seq_length by default.")
            self.host_max_kv_cache_lengths = [
                torch.ones((1,), dtype=torch.int32) * self.max_kv_cache_length for i in range(self.num_layers)
            ]
        elif isinstance(max_kv_cache_length, int):
            if max_kv_cache_length > self.max_seq_length + self.init_kv_cache_size:
                logger.warning(
                    "The value of max_kv_cache_length should ideally not exceed max_seq_length. "
                    "Therefore, it has been adjusted to match the value of max_seq_length."
                )
            self.max_kv_cache_length = min(max_kv_cache_length, self.max_seq_length + self.init_kv_cache_size)
            self.host_max_kv_cache_lengths = [
                torch.ones((1,), dtype=torch.int32) * self.max_kv_cache_length for i in range(self.num_layers)
            ]
        elif isinstance(max_kv_cache_length, torch.Tensor):
            self.max_kv_cache_length = int(torch.max(max_kv_cache_length).item())
            if self.max_kv_cache_length > self.max_seq_length + self.init_kv_cache_size:
                logger.warning(
                    "The value of max_kv_cache_length should ideally not exceed max_seq_length. "
                    "Therefore, it has been adjusted to match the value of max_seq_length."
                )
            self.max_kv_cache_length = min(self.max_kv_cache_length, self.max_seq_length + self.init_kv_cache_size)
            if max_kv_cache_length.shape[0] != self.num_layers:
                logger.error(
                    "max_kv_cache_length tensor's size is not equal to num_layers! "
                    "Note that num_layers = num_total_layers // pipeline_parallelism_size."
                )
                assert False
            self.host_max_kv_cache_lengths = [
                torch.minimum(max_kv_cache_length.to(torch.int32)[i], torch.IntTensor([self.max_seq_length]))
                for i in range(self.num_layers)
            ]
        else:
            assert False, "invalid max_kv_cache_length!"
        self.lora_manager = lora_manager

        self.buffer = {}
        if self.mapping.is_last_pp_rank():
            self.buffer["logits"] = torch.empty(
                (batch_size, self.vocab_size_padded)
                if not self.gather_all_token_logits
                else (batch_size, max_context_length, self.vocab_size_padded),
                dtype=self._tensor_dtype("logits"),
                device=self.device,
            )
        if self.cross_attention:
            # use shape info to pass max length info in remove padding mode
            self.buffer["encoder_max_input_length"] = torch.empty(
                (encoder_max_input_length,), dtype=self._tensor_dtype("encoder_max_input_length"), device=self.device
            )

        if self.paged_kv_cache:
            blocks = batch_size * beam_width * math.ceil(self.max_kv_cache_length / self.tokens_per_block)
            cache_shape = (
                blocks,
                2,
                self.num_heads_kv,
                self.tokens_per_block,
                self.head_size,
            )
        else:
            cache_shape = (
                batch_size,
                2,
                self.num_heads_kv,
                self.max_kv_cache_length,
                self.head_size,
            )
            if self.cross_attention:
                cross_cache_shape = (
                    batch_size,
                    2,
                    self.num_heads_kv,
                    self.encoder_max_input_length,
                    self.head_size,
                )

        for i in range(self.first_layer, self.last_layer):
            if self.quant_mode.has_kv_cache_quant():
                # Since torch does not support fp8 now, using int8 here.
                kv_cache_type = torch.int8
            else:
                kv_cache_type = self.dtype if self.paged_kv_cache else self._tensor_dtype(f"present_key_value_{i}")
            self.buffer[f"present_key_value_{i}"] = torch.zeros(
                [batch_size, 2, self.num_heads_kv, max_context_length + self.init_kv_cache_size, self.head_size],
                dtype=kv_cache_type,
                device=self.device,
            )

            if self.cross_attention:
                self.buffer[f"cross_present_key_value_{i}"] = torch.empty(
                    cross_cache_shape, dtype=kv_cache_type, device=self.device
                )

        if self.use_gpt_attention_plugin:
            self.sequence_length_buffer = torch.ones((batch_size,), dtype=torch.int32, device=self.device)
        else:
            # without plugin, we need two set of kv cache buffers,
            # one for inputs, and the other for outputs.
            # They will take turns to act as input and output buffers.
            # Not applicable to cross KV buffers as it's constant
            for i in range(self.first_layer, self.last_layer):
                trt_dtype = self.runtime.engine.get_tensor_dtype(f"present_key_value_{i}")
                if trt_dtype == trt.fp8:
                    # PyTorch doesn't support fp8 datatype, use int8 instead of it because int8 datatype size is same with fp8.
                    # TODO: Remove this section when PyTorch support fp8 datatype
                    dtype = torch.int8
                else:
                    dtype = self._tensor_dtype(f"present_key_value_{i}")
                self.buffer[f"1_present_key_value_{i}"] = torch.zeros(
                    [batch_size, 2, self.num_heads_kv, self.init_kv_cache_size, self.head_size],
                    dtype=dtype,
                    device=self.device,
                )

        if self.use_custom_all_reduce and self.mapping.tp_size > 1:
            set_peer_access(self.mapping)
            float_element_size = torch.tensor([], dtype=torch.float).element_size()
            buffer_size = (
                batch_size
                * beam_width
                * max_context_length
                * self.hidden_size
                * self.mapping.tp_size
                * float_element_size
            )
            barrier_size = IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * self.mapping.tp_size

            self.ipc_buffers = IpcMemory(self.mapping, buffer_size)
            self.ipc_barriers_in = IpcMemory(self.mapping, barrier_size)
            self.ipc_barriers_out = IpcMemory(self.mapping, barrier_size)
            self.all_reduce_workspace = torch.tensor(
                self.ipc_buffers.serialize() + self.ipc_barriers_in.serialize() + self.ipc_barriers_out.serialize(),
                dtype=torch.int64,
                device="cpu",
            )

        if self.use_lora_plugin and self.lora_manager is not None:
            assert lora_uids is not None
            lora_weights_pointers_list = [
                torch.zeros(size=(batch_size, 2), dtype=torch.int64).contiguous().cpu() for _ in range(self.num_layers)
            ]

            for idx in range(self.num_layers):
                layer_idx = idx + self.first_layer

                for lora_module in self.lora_target_modules:
                    self.buffer.update(
                        {
                            f"{lora_module}_lora_ranks_{layer_idx}": torch.zeros(size=(batch_size,), dtype=torch.int32)
                            .contiguous()
                            .cpu()
                        }
                    )

                    self.buffer.update(
                        {
                            f"{lora_module}_lora_weights_pointers_{layer_idx}": torch.zeros(
                                size=(batch_size, 2), dtype=torch.int64
                            )
                            .contiguous()
                            .cpu()
                        }
                    )
                    for batch_idx in range(batch_size):
                        lora_uid = lora_uids[batch_idx]
                        if lora_uid is not None and lora_uid != "-1":
                            self.buffer[f"{lora_module}_lora_ranks_{layer_idx}"][
                                batch_idx
                            ] = self.lora_manager.uid_to_low_ranks(lora_uid)[layer_idx][lora_module]

                            self.buffer[f"{lora_module}_lora_weights_pointers_{layer_idx}"][batch_idx][
                                0
                            ] = self.lora_manager.lora_weights_pointers_list[layer_idx][lora_uid][lora_module][0]
                            self.buffer[f"{lora_module}_lora_weights_pointers_{layer_idx}"][batch_idx][
                                1
                            ] = self.lora_manager.lora_weights_pointers_list[layer_idx][lora_uid][lora_module][1]
                        else:
                            self.buffer[f"{lora_module}_lora_ranks_{layer_idx}"][batch_idx] = 0

        self.buffer_allocated = True

    def _get_context_shape_buffer(
        self,
        input_ids: torch.Tensor,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        position_ids: torch.Tensor,
        last_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_indirection: torch.Tensor,
        kv_cache_block_pointers: List[torch.Tensor],
        host_kv_cache_block_pointers: List[torch.Tensor],
        hidden_states_input: torch.Tensor = None,
        prompt_embedding_table: torch.Tensor = None,
        tasks: torch.Tensor = None,
        prompt_vocab_size: torch.Tensor = None,
        encoder_output: torch.Tensor = None,
        encoder_input_lengths: torch.Tensor = None,
    ) -> List[RuntimeTensor]:
        tensors = []
        sym = lambda x, name: RuntimeTensor.from_torch(name, x)
        add_tensor = lambda x, name: tensors.append(sym(x, name))
        add_tensor_with_shape = lambda x, name, shape: tensors.append(
            RuntimeTensor.from_torch(name, x, override_shape=shape)
        )

        add_tensor(context_lengths, "context_lengths")
        add_tensor(cache_indirection, "cache_indirection")

        if self.has_position_embedding:
            add_tensor(position_ids, "position_ids")

        if self.cross_attention:
            add_tensor(encoder_output, "encoder_output")
            add_tensor(encoder_input_lengths, "encoder_input_lengths")
            add_tensor(self.buffer["encoder_max_input_length"], "encoder_max_input_length")

        if self.mapping.has_pp():
            hidden_size = self.hidden_size * self.mapping.tp_size
            hidden_states_input = hidden_states_input.resize_(input_ids.shape[0], input_ids.shape[1], hidden_size)

        if self.mapping.is_last_pp_rank():
            add_tensor(self.buffer["logits"], "logits")

            if not self.gather_all_token_logits:
                add_tensor(last_token_ids, "last_token_ids")
        else:
            add_tensor(hidden_states_input, "hidden_states_output")

        if self.mapping.is_first_pp_rank():
            add_tensor(input_ids, "input_ids")
        else:
            add_tensor(hidden_states_input, "hidden_states_input")

        if prompt_embedding_table is not None:
            add_tensor(prompt_embedding_table, "prompt_embedding_table")

            if self.remove_input_padding:
                tasks_generation = (
                    torch.concat(
                        [
                            torch.full([context_lengths[b].item()], tasks[b].item(), dtype=torch.int32)
                            for b in range(context_lengths.size(0))
                        ]
                    )
                    .unsqueeze(0)
                    .cuda()
                )
            else:
                tasks_generation = tasks.unsqueeze(-1)
            add_tensor(tasks_generation, "tasks")
            add_tensor(prompt_vocab_size, "prompt_vocab_size")

        if self.paged_kv_cache:
            for idx in range(self.num_layers):
                layer_idx = idx + self.first_layer
                buffer = kv_cache_block_pointers[idx].contiguous()
                shape = kv_cache_block_pointers[idx].shape
                shape = [shape[0] * shape[1], *shape[2:]]
                add_tensor_with_shape(buffer, f"kv_cache_block_pointers_{layer_idx}", shape)
                add_tensor_with_shape(
                    host_kv_cache_block_pointers[idx], f"host_kv_cache_block_pointers_{layer_idx}", shape
                )

        batch_size = context_lengths.shape[0]
        if not self.paged_kv_cache:
            for idx in range(self.first_layer, self.last_layer):
                if not self.use_gpt_attention_plugin:
                    kv_cache_shape = (batch_size, 2, self.num_heads_kv, self.init_kv_cache_size, self.head_size)
                    if not self.use_streaming_llm:
                        # for empty tensor, TRT does not really use the tensor data, so any dtype is fine
                        kv_cache_buffer = torch.zeros((1,), dtype=torch.float32, device=self.device)
                    else:
                        kv_cache_buffer = self.init_past_key_values[idx]
                    add_tensor_with_shape(kv_cache_buffer, f"past_key_value_{idx}", kv_cache_shape)
                    present = f"present_key_value_{idx}"
                    add_tensor(self.buffer[present], present)

                    if self.cross_attention:
                        cross_kv_cache_shape = (batch_size, 2, self.num_heads_kv, 0, self.head_size)
                        # for empty tensor, TRT does not really use the tensor data, so any dtype is fine
                        cross_kv_cache_buffer = torch.zeros((1,), dtype=torch.float32, device=self.device)
                        add_tensor_with_shape(
                            cross_kv_cache_buffer, f"cross_past_key_value_{idx}", cross_kv_cache_shape
                        )
                        cross_present = f"cross_present_key_value_{idx}"
                        add_tensor(self.buffer[cross_present], cross_present)
                else:
                    key_value_cache = self.buffer[f"present_key_value_{idx}"]
                    # when plugin is used, past_ket_value tensor does not need to be empty tensor
                    # because plugin does not care, and does not use this shape.
                    add_tensor(key_value_cache, f"past_key_value_{idx}")
                    add_tensor(key_value_cache, f"present_key_value_{idx}")

                    if self.cross_attention:
                        cross_cache_buffer = self.buffer[f"cross_present_key_value_{idx}"]
                        add_tensor(cross_cache_buffer, f"cross_past_key_value_{idx}")
                        add_tensor(cross_cache_buffer, f"cross_present_key_value_{idx}")

        if self.use_gpt_attention_plugin:
            # context request
            host_request_types = torch.zeros_like(context_lengths, device="cpu").int()
            self.sequence_length_buffer = context_lengths.detach().clone()
            add_tensor_with_shape(self.sequence_length_buffer, "sequence_length", (batch_size,))

            # field 0: past_key_value_length, field 1: is_context (deprecated). changed to [0], otherwise affects batch padded input mode
            add_tensor_with_shape(host_context_lengths, "host_past_key_value_lengths", (batch_size,))
            add_tensor(host_request_types, "host_request_types")
            for idx in range(self.first_layer, self.last_layer):
                add_tensor_with_shape(
                    self.host_max_kv_cache_lengths[idx - self.first_layer], f"host_max_kv_cache_length_{idx}", (1,)
                )
            if self.remove_input_padding:
                add_tensor(host_context_lengths, "host_context_lengths")
        else:
            add_tensor(attention_mask, "attention_mask")

        if self.use_custom_all_reduce and self.mapping.tp_size > 1:
            add_tensor(self.all_reduce_workspace, "all_reduce_workspace")

        if self.use_lora_plugin:
            for idx in range(self.num_layers):
                for lora_module in self.lora_target_modules:
                    layer_idx = idx + self.first_layer
                    lora_ranks = f"{lora_module}_lora_ranks_{layer_idx}"
                    add_tensor(self.buffer[lora_ranks], lora_ranks)
                    lora_weights = f"{lora_module}_lora_weights_pointers_{layer_idx}"
                    add_tensor(self.buffer[lora_weights], lora_weights)

        return tensors

    def _get_next_step_shape_buffer(
        self,
        batch_size: int,
        beam_width: int,
        max_context_length: int,
        step: int,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        position_ids: torch.Tensor,
        last_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_indirection: torch.Tensor,
        kv_cache_block_pointers: List[torch.Tensor],
        host_kv_cache_block_pointers: List[torch.Tensor],
        hidden_states_input: torch.Tensor = None,
        prompt_embedding_table: torch.Tensor = None,
        tasks: torch.Tensor = None,
        prompt_vocab_size: torch.Tensor = None,
        encoder_output: torch.Tensor = None,
        encoder_input_lengths: torch.Tensor = None,
    ):
        next_step_shape = {
            "context_lengths": context_lengths.shape,
            "cache_indirection": cache_indirection.shape,
        }
        next_step_buffer = {
            "context_lengths": context_lengths.contiguous(),
            "cache_indirection": cache_indirection.contiguous(),
        }

        if self.mapping.has_pp():
            hidden_size = self.hidden_size * self.mapping.tp_size
            shape = (
                (1, batch_size * beam_width, hidden_size)
                if self.remove_input_padding
                else (batch_size * beam_width, 1, hidden_size)
            )
            hidden_states_input = hidden_states_input.resize_(*shape)

        if self.mapping.is_last_pp_rank():
            next_step_buffer["logits"] = self.buffer["logits"]

            if not self.gather_all_token_logits:
                next_step_shape["last_token_ids"] = last_token_ids.shape
                next_step_buffer["last_token_ids"] = last_token_ids.contiguous()
        else:
            next_step_shape["hidden_states_output"] = hidden_states_input.shape
            next_step_buffer["hidden_states_output"] = hidden_states_input.contiguous()

        if self.mapping.is_first_pp_rank():
            next_step_shape["input_ids"] = (
                (1, batch_size * beam_width) if self.remove_input_padding else (batch_size * beam_width, 1)
            )
            next_step_buffer["input_ids"] = self.new_tokens
        else:
            next_step_shape["hidden_states_input"] = hidden_states_input.shape
            next_step_buffer["hidden_states_input"] = hidden_states_input.contiguous()

        if self.remove_input_padding:
            next_step_shape["host_context_lengths"] = host_context_lengths.shape
            next_step_buffer["host_context_lengths"] = host_context_lengths.contiguous()

        if self.has_position_embedding:
            next_step_shape["position_ids"] = position_ids.shape
            next_step_buffer["position_ids"] = position_ids.contiguous()

        if self.cross_attention:
            # hack: disable (or minimize) cross qkv computation at generation phase
            # TODO: enable [0,0,.] true zero tensor input; or use IfConditionalLayer
            next_step_shape["encoder_output"] = [1, 1, encoder_output.shape[-1]]  # encoder_output.shape
            next_step_shape["encoder_input_lengths"] = encoder_input_lengths.shape
            next_step_shape["encoder_max_input_length"] = self.buffer["encoder_max_input_length"].shape
            next_step_buffer["encoder_output"] = encoder_output.contiguous()
            next_step_buffer["encoder_input_lengths"] = encoder_input_lengths.contiguous()
            next_step_buffer["encoder_max_input_length"] = self.buffer["encoder_max_input_length"]

        if self.paged_kv_cache:
            for idx in range(self.num_layers):
                layer_idx = idx + self.first_layer
                next_step_buffer[f"kv_cache_block_pointers_{layer_idx}"] = kv_cache_block_pointers[idx].contiguous()
                next_step_buffer[f"host_kv_cache_block_pointers_{layer_idx}"] = host_kv_cache_block_pointers[
                    idx
                ].contiguous()
                shape = kv_cache_block_pointers[idx].shape
                shape = [shape[0] * shape[1], *shape[2:]]
                next_step_shape[f"kv_cache_block_pointers_{layer_idx}"] = shape
                next_step_shape[f"host_kv_cache_block_pointers_{layer_idx}"] = shape

        if prompt_embedding_table is not None:
            next_step_buffer["prompt_embedding_table"] = prompt_embedding_table.contiguous()
            next_step_shape["prompt_embedding_table"] = prompt_embedding_table.shape

            if self.remove_input_padding:
                gen_tasks = tasks.unsqueeze(0)
            else:
                gen_tasks = tasks.unsqueeze(-1)

            next_step_buffer["tasks"] = gen_tasks.contiguous()
            next_step_shape["tasks"] = gen_tasks.shape

            next_step_buffer["prompt_vocab_size"] = prompt_vocab_size.contiguous()
            next_step_shape["prompt_vocab_size"] = prompt_vocab_size.shape

        if not self.paged_kv_cache:
            for idx in range(self.first_layer, self.last_layer):
                if not self.use_gpt_attention_plugin:
                    if step % 2:
                        self.buffer[f"present_key_value_{idx}"] = self.buffer[f"1_present_key_value_{idx}"].new_zeros(
                            (
                                batch_size,
                                2,
                                self.num_heads_kv,
                                self.init_kv_cache_size + max_context_length + step + 1,
                                self.head_size,
                            )
                        )
                        next_step_buffer.update(
                            {
                                f"past_key_value_{idx}": self.buffer[f"1_present_key_value_{idx}"],
                                f"present_key_value_{idx}": self.buffer[f"present_key_value_{idx}"],
                            }
                        )
                    else:
                        self.buffer[f"1_present_key_value_{idx}"] = self.buffer[f"present_key_value_{idx}"].new_zeros(
                            (
                                batch_size,
                                2,
                                self.num_heads_kv,
                                self.init_kv_cache_size + max_context_length + step + 1,
                                self.head_size,
                            )
                        )
                        next_step_buffer.update(
                            {
                                f"past_key_value_{idx}": self.buffer[f"present_key_value_{idx}"],
                                f"present_key_value_{idx}": self.buffer[f"1_present_key_value_{idx}"],
                            }
                        )
                    next_shape = (
                        batch_size * beam_width,
                        2,
                        self.num_heads_kv,
                        self.init_kv_cache_size + max_context_length + step,
                        self.head_size,
                    )
                    next_step_shape[f"past_key_value_{idx}"] = next_shape
                else:
                    key_value_cache = self.buffer[f"present_key_value_{idx}"]
                    cache_shape = key_value_cache.shape
                    next_step_buffer.update(
                        {
                            f"past_key_value_{idx}": key_value_cache,
                            f"present_key_value_{idx}": key_value_cache,
                        }
                    )
                    next_step_shape[f"past_key_value_{idx}"] = cache_shape
                    if self.cross_attention:
                        cross_cache_shape = self.buffer[f"cross_present_key_value_{idx}"].shape
                        cross_cache_buffer = self.buffer[f"cross_present_key_value_{idx}"]
                        next_step_buffer.update(
                            {
                                f"cross_past_key_value_{idx}": cross_cache_buffer,
                                f"cross_present_key_value_{idx}": cross_cache_buffer,
                            }
                        )
                        next_step_shape[f"cross_past_key_value_{idx}"] = cross_cache_shape

        if self.use_gpt_attention_plugin:
            # generation requests
            host_request_types = torch.ones_like(context_lengths, device="cpu").int()
            # previous [past_kv_length, is_context] has been deprecated. only past_kv_length should be given here
            # Note we should use max_context_length here to align to max -- but isn't this done in attn plugin's max_element() already?
            host_past_key_value_lengths = torch.tensor(
                [max_context_length + step] * (batch_size * beam_width), dtype=torch.int32, device="cpu"
            )
            next_step_shape.update(
                {
                    "sequence_length": (batch_size * beam_width,),
                    "host_past_key_value_lengths": host_past_key_value_lengths.shape,
                    "host_request_types": host_request_types.shape,
                }
            )
            for idx in range(self.first_layer, self.last_layer):
                next_step_shape.update(
                    {
                        f"host_max_kv_cache_length_{idx}": (1,),
                    }
                )
                next_step_buffer.update(
                    {
                        f"host_max_kv_cache_length_{idx}": self.host_max_kv_cache_lengths[idx - self.first_layer],
                    }
                )
            next_step_buffer.update(
                {
                    # Sequence lengths are not used in the context phase actually.
                    "sequence_length": self.sequence_length_buffer,
                    "host_past_key_value_lengths": host_past_key_value_lengths,
                    "host_request_types": host_request_types,
                }
            )
            if self.remove_input_padding:
                next_step_buffer["host_context_lengths"] = host_context_lengths.contiguous()
                next_step_shape["host_context_lengths"] = host_context_lengths.shape
        else:
            next_step_shape.update({"attention_mask": attention_mask.shape})
            next_step_buffer.update(
                {
                    "attention_mask": attention_mask.contiguous(),
                }
            )

        if self.use_custom_all_reduce and self.mapping.tp_size > 1:
            next_step_shape["all_reduce_workspace"] = self.all_reduce_workspace.shape
            next_step_buffer["all_reduce_workspace"] = self.all_reduce_workspace

        if self.use_lora_plugin:
            for idx in range(self.num_layers):
                layer_idx = idx + self.first_layer
                for lora_module in self.lora_target_modules:
                    next_step_shape[f"{lora_module}_lora_ranks_{layer_idx}"] = self.buffer[
                        f"{lora_module}_lora_ranks_{layer_idx}"
                    ].shape
                    next_step_buffer[f"{lora_module}_lora_ranks_{layer_idx}"] = self.buffer[
                        f"{lora_module}_lora_ranks_{layer_idx}"
                    ]
                    next_step_shape[f"{lora_module}_lora_weights_pointers_{layer_idx}"] = self.buffer[
                        f"{lora_module}_lora_weights_pointers_{layer_idx}"
                    ].shape
                    next_step_buffer[f"{lora_module}_lora_weights_pointers_{layer_idx}"] = self.buffer[
                        f"{lora_module}_lora_weights_pointers_{layer_idx}"
                    ]

        return next_step_shape, next_step_buffer

    def handle_per_step(
        self,
        cache_indirections: list,
        step: int,
        batch_size: int,
        max_context_length: int,
        beam_width: int,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        scfg: SamplingConfig,
        kv_cache_block_pointers: list,
        host_kv_cache_block_pointers: list,
        prompt_embedding_table: torch.Tensor,
        tasks: torch.Tensor,
        context_lengths: torch.Tensor,
        host_context_lengths,
        attention_mask: torch.Tensor,
        prompt_vocab_size: torch.Tensor,
        ite: int,
        sequence_limit_lengths: torch.Tensor,
        sequence_lengths: torch.Tensor,
        next_step_buffer: dict,
        stop_words_list,
        bad_words_list,
        no_repeat_ngram_size,
        encoder_output: torch.Tensor,
        encoder_input_lengths: torch.Tensor,
    ):
        (
            should_stop,
            next_step_buffer,
            tasks,
            context_lengths,
            host_context_lengths,
            attention_mask,
            context_logits,
            encoder_input_lengths,
        ) = super().handle_per_step(
            cache_indirections,
            step,
            batch_size,
            max_context_length,
            beam_width,
            input_ids,
            hidden_states,
            scfg,
            kv_cache_block_pointers,
            host_kv_cache_block_pointers,
            prompt_embedding_table,
            tasks,
            context_lengths,
            host_context_lengths,
            attention_mask,
            prompt_vocab_size,
            ite,
            sequence_limit_lengths,
            sequence_lengths,
            next_step_buffer,
            stop_words_list,
            bad_words_list,
            no_repeat_ngram_size,
            encoder_output,
            encoder_input_lengths,
        )
        self.latest_buffer = next_step_buffer
        if should_stop is not None and should_stop.item() and step != self.max_new_tokens - 1:
            self.early_stopped = True
        return (
            should_stop,
            next_step_buffer,
            tasks,
            context_lengths,
            host_context_lengths,
            attention_mask,
            context_logits,
            encoder_input_lengths,
        )

    def get_present_key_values(self) -> Dict[int, torch.Tensor]:
        assert self.latest_buffer is not None and len(self.latest_buffer) > 0
        present_key_values = {}
        for i in range(self.first_layer, self.last_layer):
            # for early stopped case, we need to use past key values
            key = f"past_key_value_{i}" if self.early_stopped else f"present_key_value_{i}"
            present_key_values[i] = self.latest_buffer[key]
        return present_key_values

    def _prepare_context_inputs(
        self,
        batch_size,
        context_lengths,
        host_context_lengths,
        use_gpt_attention_plugin,
        remove_input_padding,
        **kwargs,
    ):
        inputs = super()._prepare_context_inputs(
            batch_size, context_lengths, host_context_lengths, use_gpt_attention_plugin, remove_input_padding, **kwargs
        )
        if "attention_mask" in inputs and self.init_kv_cache_size > 0:
            attention_mask = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat(
                (attention_mask.new_ones((attention_mask.size(0), self.init_kv_cache_size)), attention_mask), dim=-1
            ).contiguous()
        return inputs
