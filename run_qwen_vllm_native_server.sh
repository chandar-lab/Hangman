#!/usr/bin/env bash
set -euo pipefail

# Activate local virtualenv if present
if [ -f ./.venv/bin/activate ]; then
  source ./.venv/bin/activate
fi

# Defaults; override via env vars
MODEL="${MODEL:-Qwen/Qwen3-14B}"
PORT="${PORT:-8001}"

export HF_HOME="${HF_HOME:-$HOME/scratch}"

echo "Starting vLLM native two-pass HTTP server on port ${PORT} with model ${MODEL}" >&2
python -m hangman.providers.vllm_http_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --trust-remote-code


