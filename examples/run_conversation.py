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
# https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/examples/run.py


import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorrt_llm
import torch
from tensorrt_llm.logger import logger
from utils import (
    DEFAULT_HF_MODEL_DIRS,
    DEFAULT_PROMPT_TEMPLATES,
    load_tokenizer,
    read_model_name_from_config,
    throttle_generator,
)

from swiftinfer.runtime import ModelRunner, StartRecentKVCache


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        dest="input_file",
        type=str,
        help="JSONL file containing tokenized input. Alternative to text input. Same format as streaming-llm",
        default=None,
    )
    parser.add_argument("--max_output_len", type=int, required=True)
    parser.add_argument(
        "--max_kv_cache_length",
        type=int,
        default=None,
        help="The max kv cache length. \
              If the final sequence length exceeds the kv cache length, we will enable cyclic kv cache. \
              If it is set to None, we will use the max sequence length.",
    )
    parser.add_argument("--log_level", type=str, default="error")
    parser.add_argument("--engine_dir", type=str, default="engine_outputs")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--output_csv", type=str, help="CSV file where the tokenized output is stored.", default=None)
    parser.add_argument("--output_npy", type=str, help="Numpy file where the tokenized output is stored.", default=None)
    parser.add_argument(
        "--output_logits_npy",
        type=str,
        help="Numpy file where the generation logits are stored. Use only when num_beams==1",
        default=None,
    )
    parser.add_argument("--tokenizer_dir", help="HF tokenizer config path", default="gpt2")
    parser.add_argument("--vocab_file", help="Used for sentencepiece tokenizers")
    parser.add_argument("--num_beams", type=int, help="Use beam search if num_beams >1", default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--debug_mode", default=False, action="store_true", help="Whether or not to turn on the debug mode"
    )
    parser.add_argument(
        "--no_add_special_tokens",
        dest="add_special_tokens",
        default=True,
        action="store_false",
        help="Whether or not to add special tokens",
    )
    parser.add_argument("--streaming", default=False, action="store_true")
    parser.add_argument("--streaming_interval", type=int, help="How often to return tokens when streaming.", default=5)
    parser.add_argument("--prompt_table_path", type=str, help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument("--prompt_tasks", help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
    parser.add_argument("--lora_dir", type=str, default=None, help="The directory of LoRA weights")
    parser.add_argument(
        "--lora_task_uids",
        type=str,
        default=None,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module",
    )
    parser.add_argument("--streaming_llm_start_size", type=int, default=4)
    parser.add_argument("--only_n_first", type=int, default=None)
    parser.add_argument(
        "--prompt-template",
        choices=list(DEFAULT_PROMPT_TEMPLATES.keys()),
        default="vicuna",
        type=str,
        help="Choose which prompt template to use",
    )

    return parser.parse_args(args=args)


def parse_input(
    tokenizer, input_text=None, prompt_template=None, add_special_tokens=True, max_input_length=923, pad_id=None
):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []

    if isinstance(input_text, str):
        input_text = [input_text]

    for curr_text in input_text:
        if prompt_template is not None:
            curr_text = prompt_template.format(input_text=curr_text)
        input_ids = tokenizer.encode(
            curr_text, add_special_tokens=add_special_tokens, truncation=True, max_length=max_input_length
        )
        batch_input_ids.append(input_ids)

    batch_input_ids = [torch.tensor(x, dtype=torch.int32).unsqueeze(0) for x in batch_input_ids]
    return batch_input_ids


def print_output(
    tokenizer,
    output_ids,
    input_lengths,
    sequence_lengths,
    output_csv=None,
    output_npy=None,
    context_logits=None,
    generation_logits=None,
    output_logits_npy=None,
):
    batch_size, num_beams, _ = output_ids.size()
    if output_csv is None and output_npy is None:
        for batch_idx in range(batch_size):
            inputs = output_ids[batch_idx][0][: input_lengths[batch_idx]].tolist()
            input_text = tokenizer.decode(inputs)
            print(f'Input [Text {batch_idx}]: "{input_text}"')
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                print(f'Output [Text {batch_idx} Beam {beam}]: "{output_text}"')

    output_ids = output_ids.reshape((-1, output_ids.size(2)))

    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids.tolist()
        with open(output_file, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerows(outputs)

    if output_npy is not None:
        output_file = Path(output_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = np.array(output_ids.cpu().contiguous(), dtype="int32")
        np.save(output_file, outputs)

    if generation_logits is not None and output_logits_npy is not None and num_beams == 1:
        input_lengths = torch.Tensor(input_lengths)
        context_logits = torch.cat(context_logits, axis=0)
        generation_logits = [logit.unsqueeze(1) for logit in generation_logits]
        generation_logits = torch.cat(generation_logits, axis=1)
        last_token_ids = torch.cumsum(input_lengths, dim=0).int().cuda()
        batch_size = input_lengths.size(0)
        vocab_size_padded = context_logits.shape[-1]
        context_logits = context_logits.reshape([1, -1, vocab_size_padded])
        context_logits = torch.index_select(context_logits, 1, last_token_ids - 1).view(
            batch_size, 1, vocab_size_padded
        )
        logits = torch.cat([context_logits, generation_logits], axis=1)
        logits = logits.reshape(-1, num_beams, logits.shape[1], logits.shape[2])
        output_file = Path(output_logits_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = np.array(logits.cpu().contiguous(), dtype="float32")
        np.save(output_file, outputs)


def generate_sample(
    args,
    rank,
    tokenizer,
    batch_input_ids,
    input_lengths,
    runner,
    end_id,
    pad_id,
    past_key_values,
    stop_words_list,
    bad_words_list,
):
    outputs = runner.generate(
        batch_input_ids,
        max_new_tokens=args.max_output_len,
        max_kv_cache_length=args.max_kv_cache_length,
        end_id=end_id,
        pad_id=pad_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        stop_words_list=stop_words_list,
        bad_words_list=bad_words_list,
        lora_uids=args.lora_task_uids,
        prompt_table_path=args.prompt_table_path,
        prompt_tasks=args.prompt_tasks,
        streaming=args.streaming,
        output_sequence_lengths=True,
        return_dict=True,
        past_key_values=past_key_values,
    )
    num_generated_tokens = 0

    if rank == 0:
        if args.streaming:
            for curr_outputs in throttle_generator(outputs, args.streaming_interval):
                output_ids = curr_outputs["output_ids"]
                sequence_lengths = curr_outputs["sequence_lengths"]
                print_output(
                    tokenizer,
                    output_ids,
                    input_lengths,
                    sequence_lengths,
                    output_csv=args.output_csv,
                    output_npy=args.output_npy,
                )

                # count number of tokens generated in streaming
                num_token_streamed = 0
                batch_size, num_beams, _ = output_ids.size()
                for batch_idx in range(batch_size):
                    for beam in range(num_beams):
                        output_begin = input_lengths[batch_idx]
                        output_end = sequence_lengths[batch_idx][beam]
                        num_token_streamed += output_end - output_begin
                num_generated_tokens = max(num_generated_tokens, num_token_streamed)
        else:
            output_ids = outputs["output_ids"]
            sequence_lengths = outputs["sequence_lengths"]
            context_logits = None
            generation_logits = None
            if runner.session.gather_all_token_logits:
                context_logits = outputs["context_logits"]
                generation_logits = outputs["generation_logits"]
            print_output(
                tokenizer,
                output_ids,
                input_lengths,
                sequence_lengths,
                output_csv=args.output_csv,
                output_npy=args.output_npy,
                context_logits=context_logits,
                generation_logits=generation_logits,
                output_logits_npy=args.output_logits_npy,
            )

            # count number of tokens generated
            batch_size, num_beams, _ = output_ids.size()
            for batch_idx in range(batch_size):
                for beam in range(num_beams):
                    output_begin = input_lengths[batch_idx]
                    output_end = sequence_lengths[batch_idx][beam]
                    num_generated_tokens += output_end - output_begin
    return num_generated_tokens


@torch.no_grad()
def streaming_inference(
    args,
    runner,
    tokenizer,
    prompts,
    end_id,
    pad_id,
    stop_words_list,
    bad_words_list,
    only_n_first: Optional[int] = None,
):
    runtime_rank = tensorrt_llm.mpi_rank()

    # TODO: refactor this
    # test past key values, shape [B, 2, H, S, D]
    # TODO: make this configurable
    kv_cache_mgr = StartRecentKVCache(
        start_size=args.streaming_llm_start_size,
        recent_size=args.max_input_length + args.max_output_len - args.streaming_llm_start_size,
    )
    past_key_values = None
    total_num_generated_tokens = 0
    if only_n_first is not None:
        prompts = prompts[:only_n_first]

    torch.cuda.synchronize()
    start = time.time()

    for idx, prompt in enumerate(prompts):
        prompt_template = DEFAULT_PROMPT_TEMPLATES[args.prompt_template]
        prompt = prompt_template.format(input_text=prompt)
        batch_input_ids = parse_input(
            tokenizer=tokenizer,
            input_text=prompt,
            add_special_tokens=args.add_special_tokens,
            max_input_length=args.max_input_length,
            pad_id=pad_id,
        )
        input_lengths = [x.size(1) for x in batch_input_ids]

        if past_key_values is not None:
            past_key_values = kv_cache_mgr.evict_for_space(
                past_key_values, int(max(input_lengths)) + args.max_output_len
            )
        num_generated_token = generate_sample(
            args=args,
            rank=runtime_rank,
            tokenizer=tokenizer,
            batch_input_ids=batch_input_ids,
            input_lengths=input_lengths,
            runner=runner,
            end_id=end_id,
            pad_id=pad_id,
            past_key_values=past_key_values,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
        )
        total_num_generated_tokens += num_generated_token

        past_key_values = runner.get_present_key_values()

        print(
            f"[Step Info]: generated {num_generated_token} tokens, kv cache lenght: {next(iter(past_key_values.values())).shape[3]}"
        )

    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Total duration: {duration:.2f}s for {len(prompts)} samples")
    print(f"Throughput: {total_num_generated_tokens / duration} tokens/s")


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    model_name = read_model_name_from_config(Path(args.engine_dir) / "config.json")
    if args.tokenizer_dir is None:
        args.tokenizer_dir = DEFAULT_HF_MODEL_DIRS[model_name]

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
    )

    runner = ModelRunner.from_dir(
        engine_dir=args.engine_dir, lora_dir=args.lora_dir, rank=runtime_rank, debug_mode=args.debug_mode
    )

    # # An example to stop generation when the model generate " London" on first sentence, " eventually became" on second sentence
    # stop_words_list = [[" London"], ["eventually became"]]
    # stop_words_list = tensorrt_llm.runtime.to_word_list_format(stop_words_list, tokenizer)
    # stop_words_list = torch.Tensor(stop_words_list).to(torch.int32).to("cuda").contiguous()
    stop_words_list = None

    # # An example to prevent generating " chef" on first sentence, " eventually" and " chef before" on second sentence
    # bad_words_list = [[" chef"], [" eventually, chef before"]]
    # bad_words_list = tensorrt_llm.runtime.to_word_list_format(bad_words_list, tokenizer)
    # bad_words_list = torch.Tensor(bad_words_list).to(torch.int32).to("cuda").contiguous()
    bad_words_list = None

    # load conversation file into a list
    if not os.path.exists(args.input_file):
        raise ValueError(f"Input file {args.input_file} does not exist.")
    prompts = []
    with open(args.input_file, "r") as f:
        for line in f:
            prompts += json.loads(line)["turns"]

    # run inference
    streaming_inference(
        args=args,
        runner=runner,
        tokenizer=tokenizer,
        prompts=prompts,
        end_id=end_id,
        pad_id=pad_id,
        stop_words_list=stop_words_list,
        bad_words_list=bad_words_list,
        only_n_first=args.only_n_first,
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
