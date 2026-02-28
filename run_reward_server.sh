#!/usr/bin/env bash
# Run vLLM OpenAI API server for Llama-3.1/3.3-Nemotron-70B-Reward.
# With transformers >=4.44 the tokenizer must have a chat template; we pass one explicitly.
# For GGUF + separate tokenizer (e.g. --model GGUF --tokenizer nvidia/...), the tokenizer
# often has no template, so --chat-template is required.

set -e
PORT="${PORT:-8000}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHAT_TEMPLATE="${SCRIPT_DIR}/vllm/chat_template_llama31.jinja"

# Optional: use GGUF model + NVIDIA tokenizer (no chat template), e.g.:
#   MODEL="second-state/Llama-3.1-Nemotron-70B-Reward-HF-GGUF:Q4_K_M"
#   TOKENIZER="nvidia/Llama-3.1-Nemotron-70B-Reward"
MODEL="${MODEL:-nvidia/Llama-3.3-Nemotron-70B-Reward}"
EXTRA_ARGS=()
if [[ -n "${TOKENIZER:-}" ]]; then
  EXTRA_ARGS+=(--tokenizer "$TOKENIZER")
fi

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    "${EXTRA_ARGS[@]}" \
    --chat-template "$CHAT_TEMPLATE" \
    --dtype auto \
    --trust-remote-code \
    --served-model-name nemotron-reward \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95
