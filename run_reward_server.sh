#!/usr/bin/env bash
# Run vLLM OpenAI API server for Llama-3.3-Nemotron-70B-Reward.
# On a single 80GB GPU the full 70B BF16 model is tight on memory; use the flags below.
# For 2+ GPUs, increase --tensor-parallel-size and you can raise --max-model-len.

set -e
PORT="${PORT:-5000}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

python3 -m vllm.entrypoints.openai.api_server \
    --model "nvidia/Llama-3.3-Nemotron-70B-Reward" \
    --dtype auto \
    --trust-remote-code \
    --served-model-name nemotron-reward \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95
