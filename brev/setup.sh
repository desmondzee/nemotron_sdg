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

# Register Jupyter kernel so notebooks can use the Data Designer environment
echo "==> Registering Jupyter kernel 'Data Designer (Nemotron)'..."
uv run python -m ipykernel install --user --name=data-designer --display-name="Data Designer (Nemotron)"

echo "==> Setup complete. Use kernel 'Data Designer (Nemotron)' in Jupyter to run the demo notebook."
echo "==> Set NVIDIA_API_KEY in the instance environment to use Nemotron models (build.nvidia.com)."
