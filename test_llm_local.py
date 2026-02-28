#!/usr/bin/env python3
"""Quick test: send an OpenAI-format request to Nemotron-3-Nano (chat/reasoning).

On the Brev instance, run vLLM on port 8000, then port-forward: brev port-forward <instance> -p 8000:8000.
This script connects to localhost:8000.

Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 — chat and reasoning model (up to 1M context).
See: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
"""

import json
import re

from openai import OpenAI

# Local port that forwards to vLLM
BASE_URL = "http://localhost:8000/v1"
# Use the --served-model-name you pass to vLLM (HF example uses "model")
MODEL = "nemotron"


def test_haiku():
    """Original test: simple haiku prompt."""
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")
    prompt = "Write a haiku about GPUs."
    print(f"Sending chat request to {BASE_URL} (model={MODEL}) ...")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    content = resp.choices[0].message.content or ""
    print(f"Prompt: {prompt!r}")
    print("Response:", content)
    print("Done.")


def _parse_json_from_response(text: str) -> dict | None:
    """Strip markdown/code fences and parse first JSON object (same as generate_grm_sft_data)."""
    if not text or not str(text).strip():
        return None
    s = str(text).strip()
    for pattern in (r"```(?:json)?\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
        m = re.search(pattern, s)
        if m:
            s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start != -1:
        depth = 0
        for i, c in enumerate(s[start:], start=start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def test_grm_json_prompt():
    """Test GRM-style prompt: force JSON with chain-of-thought reasoning and score 0-1."""
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")
    system_msg = (
        "You are a scoring assistant. Reply with only one valid JSON object. "
        'Required keys: "reasoning" (one short sentence) and "score" (number 0-1). '
        "The JSON must be complete: every string in double quotes, and the object must end with }. "
        "No other text before or after the JSON."
    )
    user_msg = (
        "Score this training example from 0 to 1 (quality for a cybersecurity classifier).\n\n"
        "Example to score:\n"
        "User: Alpha -> Beta: Initiating scan. Beta -> Alpha: Acknowledged.\n"
        "Assistant: The conversation exhibits normal operational behavior. <VERDICT> BENIGN\n\n"
        "Reply with only this structure (one short sentence for reasoning, then score). "
        'Example: {"reasoning": "Brief reason here.", "score": 0.85}'
    )
    print(f"GRM test: sending to {BASE_URL} (model={MODEL}) ...")
    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 512,
    }
    # Force JSON output if the server supports it (OpenAI-compatible response_format)
    kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        err = str(e).lower()
        if "response_format" in err or "json_object" in err or "400" in err or "422" in err:
            del kwargs["response_format"]
            print("(response_format not supported, retrying without it)")
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    msg = resp.choices[0].message if resp.choices else None
    content = (getattr(msg, "content", None) or "").strip() if msg else ""
    # Nemotron reasoning models may put output in reasoning_content
    if not content and msg and hasattr(msg, "reasoning_content") and getattr(msg, "reasoning_content"):
        content = (msg.reasoning_content or "").strip()
    print("Raw response:", repr(content[:500]) if len(content) > 500 else repr(content))
    parsed = _parse_json_from_response(content)
    # If we got valid JSON with reasoning but missing score (truncation), add default score
    if parsed and "reasoning" in parsed and "score" not in parsed:
        parsed["score"] = 0.5
    if parsed and "score" in parsed:
        score = float(parsed["score"])
        score = max(0.0, min(1.0, score))
        reasoning = (parsed.get("reasoning") or "").strip() if isinstance(parsed.get("reasoning"), str) else ""
        print(f"Parsed score: {score}")
        if reasoning:
            print(f"Reasoning: {reasoning}")
        print("JSON wraps properly.")
    else:
        print("Parse failed or missing 'score'. Parsed:", parsed)
    print("GRM test done.")


def main():
    test_haiku()
    print()
    test_grm_json_prompt()


if __name__ == "__main__":
    main()
