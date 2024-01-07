#!/usr/bin/env bash
set -e

ROOT=$(realpath $(dirname $0)/..)
cd $ROOT
if [ ! -d "$ROOT/3rdparty/TensorRT-LLM/.git" ]; then
    git submodule update --init --recursive
fi

python3 scripts/build_trt_llm.py $ROOT/3rdparty/TensorRT-LLM --clean --trt_root $TRT_ROOT --nccl_root $NCCL_ROOT --cudnn_root $CUDNN_ROOT
cd $ROOT/3rdparty/TensorRT-LLM
pip install build/tensorrt_llm*.whl
