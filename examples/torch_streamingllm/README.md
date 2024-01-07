# 🪅 Benchmark Original StremaingLLM

## 🔗 Table of Contents

- [🪅 Benchmark Original StremaingLLM](#🪅-benchmark-original-stremaingllm)
    - [🔗 Table of Contents](#🔗-table-of-contents)
    - [📌 Overview](#📌-overview)
    - [🚗 Quick Start](#🚗-quick-start)

## 📌 Overview

The original [StreamingLLM](https://github.com/mit-han-lab/streaming-llm) provides a PyTorch implementation of StreamingLLM. It contains an exmaple script to showcase its generation quality. However, the script does not provide any system metrics to evaluate how fast the model can generate text. We modify the [example script](https://github.com/mit-han-lab/streaming-llm/blob/26b72ffa944c476a7a3c5efdfab6a9b49016aaac/examples/run_streaming_llama.py) to add some metrics to evaluate its performance.

## 🚗 Quick Start

You can use the following command to install StreamingLLM easily.

```bash
# create a new conda env
conda create -yn streaming python=3.8
conda activate streaming

# install torch and related deps
pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

# install streamingllm
# we fixed the commit for reproducibility
pip install git+https://github.com/mit-han-lab/streaming-llm.git@26b72ffa944c476a7a3c5efdfab6a9b49016aaac
```

You are then ready to run the benchmark script to evaluate the performance of the PyTorch version of StreamingLLM. Note than you need to replace `<model-dir>` with the actual path to the Hugging Face model repository as mentioned in the [root README](../../README.md).

```bash
python run_streaming_llama.py \
--model_name_or_path <model-dir> \
--enable_streaming \
--max_output_len 1024 \
--max_input_len 1024 \
--start_size 4 \
--only_n_first 5
```

You can tune the arguments to evaluate the performance.
- `start_size`: the number of initial tokens to retain in the window
- `max_output_len`: the maximum number of tokens to be generated
- `only_n_first`: the number of rounds of conversation to run through, you can remove this if you want to test all converstaion data.
