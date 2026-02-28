# Creating a Brev Launchable for Nemotron SDG

This repo is set up to be deployed as an **[NVIDIA Brev](https://brev.nvidia.com)** Launchable so you can run Data Designer with **Nemotron** models on GPU instances and test the deployment end-to-end.

Follow the steps below in the [Brev Launchables documentation](https://docs.nvidia.com/brev/latest/launchables.html) to create and share your Launchable.

---

## Step 1: Files and Runtime

1. Go to [brev.nvidia.com](https://brev.nvidia.com) and sign in.
2. Open the **Launchables** tab → **Create Launchable**.
3. **Code Files:** Choose **Git Repository**.
4. Enter your **repository URL** (e.g. `https://github.com/your-org/nemotron_sdg`).  
   Only **public** repositories are supported.
5. **Runtime:** Select **VM Mode** (recommended).  
   VM Mode gives you an Ubuntu 22.04 GPU VM with Docker, Python, and CUDA pre-installed; the setup script will install the rest.

---

## Step 2: Configure the Runtime

1. **Setup script:** Use the script in this repo so the VM installs dependencies and registers the Jupyter kernel.
   - **Option A:** Upload the file **`brev/setup.sh`** from this repo.
   - **Option B:** Paste the contents of **`brev/setup.sh`** into the setup script field.
2. Click **Next**.

---

## Step 3: Jupyter and Networking

1. **Jupyter Notebook Experience:** Select **Yes** so users get a one-click Jupyter link to open the code and run the demo notebook.
2. (Optional) Configure any **secure links** or **TCP/UDP ports** you need.
3. Click **Next**.

---

## Step 4: Compute Configuration

1. Select an **NVIDIA GPU** that fits your run (e.g. for Nemotron inference, a GPU with enough VRAM for the model).
2. Adjust **Disk Storage** if needed.
3. Click **Next**.

---

## Step 5: Final Review

1. **Name:** e.g. `Nemotron SDG (Data Designer)`.
2. Review the configuration and use **Preview** to see the deploy page.
3. Click **Create Launchable** to get your shareable link and deploy page.

---

## After Deployment: Testing

### Option A: Local GGUF + llama.cpp (no API key)

The setup script installs **llama-cpp-python** and **brev/helper.py**, and downloads a default GGUF model from Hugging Face (Llama-3.1-Nemotron-70B-Instruct, Q4_K_M). To run the model locally:

1. Start the OpenAI-compatible server (from repo root, with the workspace venv):
   ```bash
   cd sdg/SDG_network && uv run python ../../brev/helper.py run
   ```
   Or download only, then start the server with a specific path:
   ```bash
   uv run python brev/helper.py download
   uv run python brev/helper.py server --model-path <path-printed-by-download>
   ```
2. Point **test.py** at the server by leaving `LOCAL_MODEL_ENDPOINT` as `http://localhost:8000/v1` (default). Run from `sdg/SDG_network`: `uv run python test.py`.

Override the default model with env: `HF_GGUF_REPO_ID`, `HF_GGUF_FILENAME`, `GGUF_CACHE_DIR`. See **brev/helper.py** docstring.

### Option B: Nemotron via build.nvidia.com (API key)

1. **Set `NVIDIA_API_KEY`** in the instance environment (from [build.nvidia.com](https://build.nvidia.com)).
2. Open **Jupyter** and run **`brev/nemotron_sdg_demo.ipynb`** with kernel **"Data Designer (Nemotron)"**.

---

## Summary

| Item | Value |
|------|--------|
| **Setup script** | `brev/setup.sh` |
| **Helper (GGUF + llama.cpp)** | `brev/helper.py` — download, server, run |
| **Helper deps** | `brev/requirements.txt` (huggingface-hub, llama-cpp-python[server]) |
| **Demo notebook** | `brev/nemotron_sdg_demo.ipynb` |
| **Jupyter kernel** | Data Designer (Nemotron) |
| **Local GGUF default** | Hugging Face: Llama-3.1-Nemotron-70B-Instruct-HF-GGUF (Q4_K_M) |
| **Env for Nemotron API** | `NVIDIA_API_KEY` (from build.nvidia.com) |
| **Docs** | [Brev Launchables](https://docs.nvidia.com/brev/latest/launchables.html) |

Once the Launchable is created, you can share the link, toggle access, view metrics, and manage it from the **Launchables** tab.
