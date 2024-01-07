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
# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.6.0/tensorrt_llm/runtime/model_runner.py


import copy
from pathlib import Path
from typing import Dict, List, Optional, Union

import tensorrt_llm
import tensorrt_llm.profiler as profiler
import torch
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import LoraManager, SamplingConfig
from tensorrt_llm.runtime.model_runner import ModelRunner as _ModelRunner
from tensorrt_llm.runtime.model_runner import get_engine_name, read_config

from .generation import GenerationSession


class ModelRunner(_ModelRunner):
    def __init__(
        self,
        session: GenerationSession,
        max_batch_size: int,
        max_input_len: int,
        lora_manager: Optional[LoraManager] = None,
    ) -> None:
        self.session = session
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.lora_manager = lora_manager

    @classmethod
    def from_dir(
        cls, engine_dir: str, lora_dir: Optional[str] = None, rank: int = 0, debug_mode: bool = False
    ) -> "ModelRunner":
        """
        Create a ModelRunner instance from an engine directory.

        Args:
            engine_dir (str):
                The directory that contains the serialized engine files and config files.
            lora_dir (str):
                The directory that contains LoRA weights.
            rank (int):
                The runtime rank id.
            debug_mode (int):
                Whether or not to turn on the debug mode.
        Returns:
            ModelRunner: An instance of ModelRunner.
        """
        # session setup
        engine_dir = Path(engine_dir)
        config_path = engine_dir / "config.json"
        model_config, other_config = read_config(config_path)
        world_size = other_config.pop("world_size")
        tp_size = other_config.pop("tp_size")
        pp_size = other_config.pop("pp_size")
        runtime_mapping = tensorrt_llm.Mapping(world_size=world_size, rank=rank, tp_size=tp_size, pp_size=pp_size)
        torch.cuda.set_device(rank % runtime_mapping.gpus_per_node)

        engine_name = get_engine_name(model_config.model_name, model_config.dtype, tp_size, pp_size, rank)
        serialize_path = engine_dir / engine_name

        profiler.start("load tensorrt_llm engine")
        with open(serialize_path, "rb") as f:
            engine_buffer = f.read()

        if model_config.model_name in ("chatglm_6b", "glm_10b", "qwen"):
            raise NotImplementedError("GenerationSession is not implemented for chatglm_6b, glm_10b, qwen.")
        else:
            session_cls = GenerationSession
        session = session_cls(model_config, engine_buffer, runtime_mapping, debug_mode=debug_mode)
        profiler.stop("load tensorrt_llm engine")
        loading_time = profiler.elapsed_time_in_sec("load tensorrt_llm engine")
        logger.info(f"Load engine takes: {loading_time} sec")

        if session.use_lora_plugin:
            assert lora_dir is not None, "lora_dir should not be None for engine built with lora_plugin enabled."
            lora_manager = LoraManager()
            lora_manager.load_from_hf(model_dir=lora_dir, model_config=model_config, runtime_mapping=runtime_mapping)
        else:
            lora_manager = None

        return cls(session, lora_manager=lora_manager, **other_config)

    def generate(
        self,
        batch_input_ids: List[torch.Tensor],
        sampling_config: Optional[SamplingConfig] = None,
        prompt_table_path: Optional[str] = None,
        prompt_tasks: Optional[str] = None,
        lora_uids: Optional[list] = None,
        streaming: bool = False,
        past_key_values: Optional[Dict[int, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, dict]:
        """
        Generates sequences of token ids.
        The generation-controlling parameters are set in the sampling_config; it will be set to a default one if not passed.
        You can override any sampling_config's attributes by passing corresponding parameters.

        Args:
            batch_input_ids (List[torch.Tensor]):
                A list of input id tensors. Each tensor is of shape (sequence_length, ).
            sampling_config (Optional[SamplingConfig]):
                The sampling configuration to be used as base parametrization for the generation call.
                The passed **kwargs matching the sampling_config's attributes will override them.
                If the sampling_config is not provided, a default will be used.
            prompt_table_path (str):
                The file path of prompt table (.npy format, exported by nemo_prompt_convert.py).
            prompt_tasks (str):
                The prompt tuning task ids for the input batch, in format of comma-separated list (e.g., 0,3,1,0).
            lora_uids (list):
                The uids of LoRA weights for the input batch. Use -1 to disable the LoRA module.
            kwargs (Dict[str, Any]:
                Ad hoc parametrization of sampling_config.
                The passed **kwargs matching the sampling_config's attributes will override them.
        Returns:
            torch.Tensor or dict:
                If return_dict=False, the method returns generated output_ids.
                If return_dict=True, the method returns a dict of output_ids,
                sequence_lengths (if sampling_config.output_sequence_lengths=True),
                context_logits and generation_logits (if self.session.gather_all_token_logits=True).
        """
        # Use sampling_config like HF's generation_config
        if sampling_config is None:
            sampling_config = SamplingConfig(end_id=None, pad_id=None)
        else:
            sampling_config = copy.deepcopy(sampling_config)
        sampling_config.update(**kwargs)

        batch_size = len(batch_input_ids)
        batch_input_ids, input_lengths = self._prepare_inputs(batch_input_ids, sampling_config.pad_id)

        if self.use_lora_plugin:
            assert lora_uids is not None, "lora_uids should not be None for engine built with lora_plugin enabled."
        self.session.setup(
            batch_size=batch_size,
            max_context_length=input_lengths.max().item(),
            max_new_tokens=sampling_config.max_new_tokens,
            beam_width=sampling_config.num_beams,
            max_kv_cache_length=sampling_config.max_kv_cache_length,
            lora_manager=self.lora_manager,
            lora_uids=lora_uids,
            past_key_values=past_key_values,
        )

        batch_input_ids = batch_input_ids.cuda()
        input_lengths = input_lengths.cuda()
        ptuning_kwargs = self._prepare_ptuning(prompt_table_path, prompt_tasks, batch_size)
        outputs = self.session.decode(
            batch_input_ids,
            input_lengths,
            sampling_config,
            stop_words_list=sampling_config.stop_words_list,
            bad_words_list=sampling_config.bad_words_list,
            output_sequence_lengths=sampling_config.output_sequence_lengths,
            return_dict=sampling_config.return_dict,
            streaming=streaming,
            **ptuning_kwargs,
        )
        if sampling_config.return_dict:
            if streaming:
                outputs = (self._prepare_outputs(curr_outputs, input_lengths) for curr_outputs in outputs)
            else:
                outputs = self._prepare_outputs(outputs, input_lengths)
        return outputs

    def get_present_key_values(self) -> Dict[int, torch.Tensor]:
        return self.session.get_present_key_values()
