#!/usr/bin/env bash
set -euo pipefail

# Activate local virtualenv if present
if [ -f ./.venv/bin/activate ]; then
  source ./.venv/bin/activate
fi

# Defaults; override via env vars
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
PORT="${PORT:-8001}"

export HF_HOME="${HF_HOME:-$HOME/scratch}"

echo "Starting vLLM native two-pass HTTP server on port ${PORT} with model ${MODEL} (dtype=${DTYPE:-half})" >&2
python -m hangman.providers.vllm_http_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  # --dtype "${DTYPE:-half}" \
  --trust-remote-code



