#!/usr/bin/env bash
set -euo pipefail

# Config-driven launcher (local machine). The caller exports the config env
# before invoking this (KUNLUN_FUSE_AR_RMSNORM / BKCL_AR_RMSNORM_MODE /
# XCCL_MESH_ALGO / KUNLUN_FUSE_AR_RMSNORM_MIN|MAX_TOKENS). TP4, fp16.

source /root/miniconda/etc/profile.d/conda.sh
conda activate python310_torch29_cuda

export LD_LIBRARY_PATH=/home/wjs/baidu/xpu/bkcl/output/so:${LD_LIBRARY_PATH:-}
export XPU_VISIBLE_DEVICES=${DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES=${DEVICES:-0,1,2,3}
export VLLM_USE_V1=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_CUDNN_ENABLED=1
export XFT_USE_FAST_SWIGLU=1
export no_proxy=127.0.0.1,localhost,${no_proxy:-}
export NO_PROXY=127.0.0.1,localhost,${NO_PROXY:-}

# Fusion off by default; the driver overrides per config.
export KUNLUN_FUSE_AR_RMSNORM=${KUNLUN_FUSE_AR_RMSNORM:-0}

python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port ${PORT:-8866} \
  --model /home/wjs/models/Qwen3-32B \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --max-model-len 32768 \
  --distributed-executor-backend mp \
  --enforce-eager
