#!/usr/bin/env bash
# Driver: model throughput comparison on TP4 / fp16 for 4 configs:
#   fused_mesh, fused_ring, baseline, mesh_baseline
# Concurrency 1..256 (x2). Launches each server, benches, stops, writes CSV.
set -uo pipefail
cd "$(dirname "$0")"

export PORT=${PORT:-8866}
export DEVICES=${DEVICES:-0,1,2,3}
ROOT=${ROOT:-/home/wjs/test/model_tp4_fp16_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$ROOT"
CONC="${CONC:-1 2 4 8 16 32 64 128 256}"

wait_ready() {
  for _ in $(seq 1 120); do
    curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}
stop_server() {
  # Kill the API server AND its worker subprocesses; SIGKILL alone leaves XPU
  # memory held until the children actually exit, so we then poll xpu-smi until
  # device memory is released before returning (else the next server sees
  # "Free memory ... less than desired GPU memory utilization").
  pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  pkill -9 -f 'VLLM::' 2>/dev/null || true
  pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true
  pkill -9 -f 'vllm' 2>/dev/null || true
  for _ in $(seq 1 60); do
    ss -ltn 2>/dev/null | grep -q ":${PORT} " && { sleep 3; continue; }
    local used
    used=$(xpu-smi 2>/dev/null | grep -oE '[0-9]+MiB / 98304MiB' | grep -oE '^[0-9]+' | sort -rn | head -1)
    used=${used:-0}
    [ "$used" -lt 3000 ] && { sleep 3; return 0; }
    sleep 3
  done
  return 0
}

run_config() {
  local name="$1"; shift
  echo "===== CONFIG $name ====="
  unset KUNLUN_FUSE_AR_RMSNORM BKCL_AR_RMSNORM_MODE XCCL_MESH_ALGO \
        KUNLUN_FUSE_AR_RMSNORM_MIN_TOKENS KUNLUN_FUSE_AR_RMSNORM_MAX_TOKENS
  for kv in "$@"; do export "$kv"; done
  local sdir="$ROOT/$name"
  mkdir -p "$sdir"
  stop_server
  bash launch.sh > "$sdir/server.log" 2>&1 &
  if ! wait_ready; then
    echo "$name: server NOT ready"; tail -30 "$sdir/server.log"; stop_server; return 1
  fi
  NUM_BATCH_LIST="$CONC" LOG_DIR="$sdir/logs" bash benchmark.sh > "$sdir/bench.log" 2>&1
  stop_server
  ( cd "$sdir/logs" && python /home/wjs/vLLM-Kunlun/tests/fused_ar_rmsnorm/to_csv.py >/dev/null \
      && cp summary.csv "$ROOT/${name}.csv" )
  echo "$name done -> $ROOT/${name}.csv"
}

CONFIGS="${CONFIGS:-fused_mesh fused_ring baseline mesh_baseline}"

want() { case " $CONFIGS " in *" $1 "*) return 0;; *) return 1;; esac; }

want fused_mesh && run_config fused_mesh    KUNLUN_FUSE_AR_RMSNORM=1 BKCL_AR_RMSNORM_MODE=rs_norm_2ag \
                         KUNLUN_FUSE_AR_RMSNORM_MIN_TOKENS=1 KUNLUN_FUSE_AR_RMSNORM_MAX_TOKENS=256
want fused_ring && run_config fused_ring    KUNLUN_FUSE_AR_RMSNORM=1 BKCL_AR_RMSNORM_MODE=ring_rs_norm_2ag \
                         KUNLUN_FUSE_AR_RMSNORM_MIN_TOKENS=1 KUNLUN_FUSE_AR_RMSNORM_MAX_TOKENS=256
want baseline && run_config baseline      KUNLUN_FUSE_AR_RMSNORM=0
want mesh_baseline && run_config mesh_baseline KUNLUN_FUSE_AR_RMSNORM=0 XCCL_MESH_ALGO=1

echo "ALL CONFIGS DONE. results root: $ROOT"
echo "$ROOT" > /home/wjs/test/last_model_tp4_fp16_root.txt
