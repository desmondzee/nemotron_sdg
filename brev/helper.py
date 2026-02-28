# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download GGUF models from Hugging Face and run them with llama.cpp (llama-cpp-python).

Usage:
  python helper.py download [--repo REPO] [--file FILE] [--cache-dir DIR]
  python helper.py server [--model-path PATH] [--host HOST] [--port PORT]
  python helper.py run   # download (if needed) and start server

Environment:
  HF_GGUF_REPO_ID   - Hugging Face repo (default: lmstudio-community/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF)
  HF_GGUF_FILENAME  - GGUF filename (default: Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf)
  GGUF_CACHE_DIR    - Where to cache downloads (default: ~/.cache/gguf or HF_HOME)
  HF_TOKEN          - Hugging Face token for gated models (optional)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Default: single-file Q4_K_M quantization of Llama-3.1-Nemotron-70B-Instruct
DEFAULT_REPO_ID = os.environ.get(
    "HF_GGUF_REPO_ID",
    "lmstudio-community/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF",
)
DEFAULT_FILENAME = os.environ.get(
    "HF_GGUF_FILENAME",
    "Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf",
)
DEFAULT_CACHE_DIR = os.environ.get("GGUF_CACHE_DIR") or os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "gguf")


def download_gguf(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
) -> str:
    """Download a GGUF model from Hugging Face and return the local path.

    Args:
        repo_id: Hugging Face repo (e.g. org/model-name).
        filename: Exact GGUF filename. If None, uses DEFAULT_FILENAME.
        cache_dir: Directory to cache the file. Defaults to GGUF_CACHE_DIR / HF_HOME.
        token: HF token for gated repos (or set HF_TOKEN).

    Returns:
        Absolute path to the downloaded .gguf file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("Install huggingface_hub: pip install huggingface-hub") from e

    filename = filename or DEFAULT_FILENAME
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    token = token or os.environ.get("HF_TOKEN")

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
    return str(Path(path).resolve())


def get_llama(model_path: str, **kwargs: object):
    """Return a llama-cpp-python Llama instance for the given GGUF path.

    Optional kwargs are passed to llama_cpp.Llama (e.g. n_ctx, n_gpu_layers).
    """
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "Install llama-cpp-python: pip install llama-cpp-python (use CMAKE_ARGS=-DGGML_CUDA=on for GPU)"
        ) from e

    return Llama(model_path=model_path, **kwargs)


def start_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    n_gpu_layers: int = -1,
) -> subprocess.Popen:
    """Start the llama-cpp-python OpenAI-compatible server in a subprocess.

    Returns the Popen instance. Call .wait() or .terminate() as needed.
    """
    # OpenAI-compatible server: python -m llama_cpp.server --model PATH --host HOST --port PORT
    cmd = [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--n_gpu_layers",
        str(n_gpu_layers),
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="GGUF download and llama.cpp server helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download
    dl = subparsers.add_parser("download", help="Download GGUF from Hugging Face")
    dl.add_argument("--repo", default=DEFAULT_REPO_ID, help="Hugging Face repo id")
    dl.add_argument("--file", default=DEFAULT_FILENAME, help="GGUF filename")
    dl.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory")
    dl.set_defaults(func=_cmd_download)

    # server
    srv = subparsers.add_parser("server", help="Start OpenAI-compatible server")
    srv.add_argument("--model-path", required=True, help="Path to .gguf model")
    srv.add_argument("--host", default="0.0.0.0", help="Bind host")
    srv.add_argument("--port", type=int, default=8000, help="Port")
    srv.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1 = all)")
    srv.set_defaults(func=_cmd_server)

    # run: download then server
    run = subparsers.add_parser("run", help="Download (if needed) and start server")
    run.add_argument("--repo", default=DEFAULT_REPO_ID, help="Hugging Face repo id")
    run.add_argument("--file", default=DEFAULT_FILENAME, help="GGUF filename")
    run.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory")
    run.add_argument("--host", default="0.0.0.0", help="Server bind host")
    run.add_argument("--port", type=int, default=8000, help="Server port")
    run.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1 = all)")
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args()
    return args.func(args)


def _cmd_download(args: argparse.Namespace) -> int:
    print(f"Downloading {args.repo} / {args.file} -> {args.cache_dir}")
    path = download_gguf(repo_id=args.repo, filename=args.file, cache_dir=args.cache_dir)
    print(f"Saved to: {path}")
    return 0


def _cmd_server(args: argparse.Namespace) -> int:
    if not Path(args.model_path).exists():
        print(f"Model not found: {args.model_path}", file=sys.stderr)
        return 1
    print(f"Starting server on {args.host}:{args.port} (model: {args.model_path})")
    proc = start_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        n_gpu_layers=args.n_gpu_layers,
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
    return proc.returncode or 0


def _cmd_run(args: argparse.Namespace) -> int:
    path = download_gguf(repo_id=args.repo, filename=args.file, cache_dir=args.cache_dir)
    print(f"Starting server on {args.host}:{args.port}")
    proc = start_server(
        model_path=path,
        host=args.host,
        port=args.port,
        n_gpu_layers=args.n_gpu_layers,
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
    return proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
