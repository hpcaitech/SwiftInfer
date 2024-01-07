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
# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/examples/utils.py


import json
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, T5Tokenizer

DEFAULT_HF_MODEL_DIRS = {
    "llama": "meta-llama/Llama-2-7b-hf",
    "vicuna": "lmsys/vicuna-7b-v1.3",
}

DEFAULT_PROMPT_TEMPLATES = {
    "llama": "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n{input_text}[/INST]",
    "vicuna": "USER: {input_text}\n\nASSISTANT: ",
}


def read_model_name_from_config(config_path: Path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["builder_config"]["name"]


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def load_tokenizer(tokenizer_dir: Optional[str] = None, vocab_file: Optional[str] = None, model_name: str = "gpt"):
    if vocab_file is None:
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, legacy=False, padding_side="left", truncation_side="left", trust_remote_code=True
        )
    else:
        # For gpt-next, directly load from tokenizer.model
        assert model_name == "gpt"
        tokenizer = T5Tokenizer(vocab_file=vocab_file, padding_side="left", truncation_side="left")

    if model_name == "qwen":
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        chat_format = gen_config["chat_format"]
        if chat_format == "raw":
            pad_id = gen_config["pad_token_id"]
            end_id = gen_config["eos_token_id"]
        elif chat_format == "chatml":
            pad_id = tokenizer.im_end_id
            end_id = tokenizer.im_end_id
        else:
            raise Exception(f"unknown chat format: {chat_format}")
    elif model_name == "glm_10b":
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id
