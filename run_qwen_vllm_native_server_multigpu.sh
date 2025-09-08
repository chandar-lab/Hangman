#!/usr/bin/env bash
set -euo pipefail

# Activate local virtualenv if present
if [ -f ./.venv/bin/activate ]; then
  source ./.venv/bin/activate
fi

# Defaults; override via env vars
MODEL="${MODEL:-Qwen/Qwen3-14B}"
GPU_LIST_CSV="${GPU_LIST:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-8001}"
DTYPE="${DTYPE:-}"

export HF_HOME="${HF_HOME:-$HOME/scratch}"

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST_CSV}"

PIDS=()
PORTS=()

cleanup() {
  echo "Stopping vLLM servers..." >&2
  for pid in "${PIDS[@]:-}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" || true
    fi
  done
  # Wait for all to exit
  for pid in "${PIDS[@]:-}"; do
    if [ -n "${pid}" ]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT TERM INT

echo "Launching ${#GPU_IDS[@]} vLLM native two-pass servers for model ${MODEL}" >&2
for idx in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$idx]}"
  port=$(( BASE_PORT + idx ))
  echo "  - GPU ${gpu} -> port ${port}" >&2
  PORTS+=("${port}")
  if [ -n "${DTYPE}" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m hangman.providers.vllm_http_server \
      --model "${MODEL}" \
      --port "${port}" \
      --dtype "${DTYPE}" \
      --trust-remote-code \
      >/dev/null 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m hangman.providers.vllm_http_server \
      --model "${MODEL}" \
      --port "${port}" \
      --trust-remote-code \
      >/dev/null 2>&1 &
  fi
  PIDS+=("$!")
done

echo "Launched PIDs: ${PIDS[*]}" >&2

# Block until all servers exit (caller backgrounds this script)
wait

#!/usr/bin/env bash
set -euo pipefail

#!/usr/bin/env bash
set -euo pipefail

# Activate local virtualenv if present
if [ -f ./.venv/bin/activate ]; then
  source ./.venv/bin/activate
fi

# Defaults; override via env vars
MODEL="${MODEL:-Qwen/Qwen3-14B}"
GPU_LIST_CSV="${GPU_LIST:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-8001}"
DTYPE="${DTYPE:-}"

export HF_HOME="${HF_HOME:-$HOME/scratch}"

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST_CSV}"

PIDS=()
PORTS=()

cleanup() {
  echo "Stopping vLLM servers..." >&2
  for pid in "${PIDS[@]:-}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" || true
    fi
  done
  # Wait for all to exit
  for pid in "${PIDS[@]:-}"; do
    if [ -n "${pid}" ]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT TERM INT

echo "Launching ${#GPU_IDS[@]} vLLM native two-pass servers for model ${MODEL}" >&2
for idx in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$idx]}"
  port=$(( BASE_PORT + idx ))
  echo "  - GPU ${gpu} -> port ${port}" >&2
  PORTS+=("${port}")
  if [ -n "${DTYPE}" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m hangman.providers.vllm_http_server \
      --model "${MODEL}" \
      --port "${port}" \
      --dtype "${DTYPE}" \
      --trust-remote-code \
      >/dev/null 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m hangman.providers.vllm_http_server \
      --model "${MODEL}" \
      --port "${port}" \
      --trust-remote-code \
      >/dev/null 2>&1 &
  fi
  PIDS+=("$!")
done

echo "Launched PIDs: ${PIDS[*]}" >&2

# Block until all servers exit (caller backgrounds this script)
wait


