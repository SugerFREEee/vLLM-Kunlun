#!/usr/bin/env bash
set -euo pipefail

# Benchmark client for the service started by:
#   /home/yzy/start_qwen3_32b_vllm_kunlun.sh
#
# The server is expected to listen on 127.0.0.1:8806 and serve model name
# "Qwen3-32B". This script only sends benchmark traffic; it does not start or
# stop the server.

source /root/miniconda/etc/profile.d/conda.sh
conda activate python310_torch29_cuda

export LD_LIBRARY_PATH=/home/wjs/baidu/xpu/bkcl/output/so:${LD_LIBRARY_PATH:-}
export no_proxy=127.0.0.1,localhost,${no_proxy:-}
export NO_PROXY=127.0.0.1,localhost,${NO_PROXY:-}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8866}
MODEL=${MODEL:-Qwen3-32B}
TOKENIZER_PATH=${TOKENIZER_PATH:-/home/wjs/models/Qwen3-32B}
MODEL_NAME=${MODEL_NAME:-qwen3_32b}

# Decode-heavy cases for AR+Residual+RMSNorm fusion attribution.
# Keep input moderate and output long enough so TPOT/output throughput reflects
# decode-side behavior. Concurrency approximates active decode M.
INPUT_LIST=(${INPUT_LIST:-128})
OUTPUT_LIST=(${OUTPUT_LIST:-256})
NUM_BATCH_LIST=(${NUM_BATCH_LIST:-1 2 4 8 16 32 64 128 256})
NUM_PROMPTS_PER_BATCH=${NUM_PROMPTS_PER_BATCH:-16}

# Optional non-4-aligned cases to verify behavior after removing m % 4 gating:
#   NUM_BATCH_LIST="9 17 33 63" bash /home/yzy/benchmark/qwen3.sh

DATE=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-${MODEL_NAME}_port${PORT}_benchmark_logs_${DATE}}
mkdir -p "${LOG_DIR}"

curl --noproxy '*' -fsS "http://${HOST}:${PORT}/v1/models" >"${LOG_DIR}/models.json"
echo "Saved served models to ${LOG_DIR}/models.json"

for INPUT in "${INPUT_LIST[@]}"; do
  for OUTPUT in "${OUTPUT_LIST[@]}"; do
    for BATCH in "${NUM_BATCH_LIST[@]}"; do
      LOG_FILE="${LOG_DIR}/benchmark_in${INPUT}_out${OUTPUT}_b${BATCH}_n${BATCH}.log"
      echo "$(date +%Y%m%d_%H%M%S): Running benchmark: input-len=${INPUT}, output-len=${OUTPUT}, max-concurrency=${BATCH}, num-prompts=${BATCH} -> ${LOG_FILE}"

      vllm bench serve \
        --host "${HOST}" \
        --port "${PORT}" \
        --backend vllm \
        --model "${MODEL}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --trust-remote-code \
        --dataset-name random \
        --num-prompts "${BATCH}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --random-input-len "${INPUT}" \
        --random-output-len "${OUTPUT}" \
        --max-concurrency "${BATCH}" \
        --request-rate inf \
        --ignore-eos \
        >"${LOG_FILE}" 2>&1

      echo "benchmark_in${INPUT}_out${OUTPUT}_b${BATCH}_n${BATCH} done."
      echo "========================================="
    done
  done
done

echo "All benchmark logs are in ${LOG_DIR}"
