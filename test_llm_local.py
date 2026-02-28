#!/usr/bin/env python3
"""Quick test: send an OpenAI-format request to the reward model.

On the Brev instance, run vLLM on port 8000, then port-forward: brev port-forward <instance> -p 8000:8000.
This script connects to localhost:8000.

Model: nvidia/Llama-3.3-Nemotron-70B-Reward — scores the quality of an assistant
response given a user prompt (higher score = higher quality).
"""

from openai import OpenAI

# Local port that forwards to vLLM (use 8000 for reward model)
BASE_URL = "http://localhost:8000/v1"
# Use the --served-model-name you pass to vLLM (e.g. nemotron-reward)
MODEL = "nemotron-reward"

def main():
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")
    # Reward model input: user prompt + assistant response to score
    prompt = "What is 1+1?"
    assistant_response = "1+1=2"
    print(f"Sending reward request to {BASE_URL} (model={MODEL}) ...")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_response},
        ],
        max_tokens=8,
    )
    content = resp.choices[0].message.content
    print(f"Prompt: {prompt!r}")
    print(f"Assistant: {assistant_response!r}")
    print("Reward (score):", content)
    print("Done.")

if __name__ == "__main__":
    main()
