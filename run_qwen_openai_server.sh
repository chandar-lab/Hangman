#!/usr/bin/env bash
set -euo pipefail

# Activate local virtualenv if present
if [ -f ./.venv/bin/activate ]; then
  source ./.venv/bin/activate
fi

# Model and port can be overridden via env vars
MODEL="${MODEL:-Qwen/Qwen3-14B}"
PORT="${PORT:-8000}"

export HF_HOME="${HF_HOME:-$HOME/scratch}"

echo "Starting vLLM OpenAI-compatible server on port ${PORT} with model ${MODEL}" >&2
python -m vllm.entrypoints.openai.api_server \
     --model "${MODEL}" \
     --trust-remote-code \
     --port "${PORT}" \
     --dtype bfloat16 \
     --enable-auto-tool-choice \
     --tool-call-parser hermes


