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

## After Deployment: Testing with Nemotron

1. **Deploy** the Launchable (e.g. **Deploy Launchable** → **Go to Instance Page**).
2. **Set `NVIDIA_API_KEY`** in the instance environment (Brev often allows env vars on the instance or in the deploy form).  
   Get an API key from [build.nvidia.com](https://build.nvidia.com) for Nemotron (e.g. `nvidia/nemotron-3-nano-30b-a3b`).
3. Open **Jupyter** from the instance page.
4. In Jupyter, open **`brev/nemotron_sdg_demo.ipynb`**.
5. Select the kernel **"Data Designer (Nemotron)"** (registered by `brev/setup.sh`).
6. Run all cells. The notebook runs a minimal Data Designer job using **Nemotron** (text and optional vision) so you can confirm the deployment works.

---

## Summary

| Item | Value |
|------|--------|
| **Setup script** | `brev/setup.sh` |
| **Demo notebook** | `brev/nemotron_sdg_demo.ipynb` |
| **Jupyter kernel** | Data Designer (Nemotron) |
| **Env var for Nemotron** | `NVIDIA_API_KEY` (from build.nvidia.com) |
| **Docs** | [Brev Launchables](https://docs.nvidia.com/brev/latest/launchables.html) |

Once the Launchable is created, you can share the link, toggle access, view metrics, and manage it from the **Launchables** tab.
