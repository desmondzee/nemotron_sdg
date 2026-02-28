#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Brev Launchable setup script for Nemotron SDG (Data Designer).
# Run on the VM after the repo is cloned to install dependencies and register the Jupyter kernel.
set -euo pipefail

echo "==> Nemotron SDG (Data Designer) – Brev setup"

# Find the Data Designer workspace (sdg/SDG_network) from repo root
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
if [[ -d "${REPO_ROOT}/sdg/SDG_network" ]]; then
    SDG_ROOT="${REPO_ROOT}/sdg/SDG_network"
elif [[ -f "${REPO_ROOT}/pyproject.toml" ]] && grep -q "data-designer-workspace" "${REPO_ROOT}/pyproject.toml" 2>/dev/null; then
    SDG_ROOT="${REPO_ROOT}"
else
    echo "ERROR: Could not find sdg/SDG_network. REPO_ROOT=${REPO_ROOT}"
    exit 1
fi

cd "${SDG_ROOT}"
echo "==> Using workspace: ${SDG_ROOT}"

# Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# Install all workspace packages + notebook dependencies (for Jupyter kernel)
echo "==> Syncing dependencies (workspace + notebooks)..."
uv sync --group notebooks

# Install Brev helper deps: GGUF download (huggingface-hub) + llama.cpp runtime (llama-cpp-python server)
echo "==> Installing llama-cpp-python and huggingface-hub for brev/helper.py..."
if command -v nvcc &>/dev/null; then
    echo "==> CUDA detected; building llama-cpp-python with GPU support..."
    export CMAKE_ARGS="-DGGML_CUDA=on"
    export FORCE_CMAKE=1
fi
uv pip install -r "${REPO_ROOT}/brev/requirements.txt" || {
    echo "==> GPU build failed or skipped; trying CPU-only llama-cpp-python..."
    unset CMAKE_ARGS FORCE_CMAKE
    uv pip install "huggingface-hub>=0.20.0" "llama-cpp-python[server]>=0.3.0"
}

# Pull default GGUF model from Hugging Face (used by helper.py server / test.py local endpoints)
echo "==> Downloading default GGUF model from Hugging Face (may take a while)..."
uv run python "${REPO_ROOT}/brev/helper.py" download || {
    echo "==> Warning: GGUF download failed. You can run later: uv run python brev/helper.py download"
}

# Register Jupyter kernel so notebooks can use the Data Designer environment
echo "==> Registering Jupyter kernel 'Data Designer (Nemotron)'..."
uv run python -m ipykernel install --user --name=data-designer --display-name="Data Designer (Nemotron)"

echo "==> Setup complete."
echo "    - Use kernel 'Data Designer (Nemotron)' in Jupyter for the demo notebook."
echo "    - To start the local GGUF server: uv run python brev/helper.py run   (or: server --model-path <path>)"
echo "    - Optional: set NVIDIA_API_KEY for Nemotron on build.nvidia.com."
