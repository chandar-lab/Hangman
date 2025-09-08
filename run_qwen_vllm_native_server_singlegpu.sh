#!/usr/bin/env bash
set -euo pipefail

# Activate local virtualenv if present
if [ -f ./.venv/bin/activate ]; then
  source ./.venv/bin/activate
fi

# Defaults; override via env vars
MODEL="${MODEL:-Qwen/Qwen3-14B}"
PORT="${PORT:-8001}"
GPU_ID="${GPU_ID:-0}"
DTYPE="${DTYPE:-}"

export HF_HOME="${HF_HOME:-$HOME/scratch}"

echo "Starting single-GPU vLLM native two-pass HTTP server on port ${PORT} with model ${MODEL} on GPU ${GPU_ID}" >&2

if [ -n "${DTYPE}" ]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m hangman.providers.vllm_http_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    --trust-remote-code
else
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m hangman.providers.vllm_http_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --trust-remote-code
fi



