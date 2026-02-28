#!/usr/bin/env python3
"""Reward scoring server using the HF forward pass (true scalar).

Llama-3.1-Nemotron-70B-Reward outputs the reward as the logit of the first output
token (index 0), not as decoded text. This script loads the Hugging Face model,
runs a single forward pass with max_new_tokens=1 and output_scores=True, and
returns scores[0][0][0].item() as the reward.

Usage:
  pip install torch transformers flask
  REWARD_MODEL=nvidia/Llama-3.1-Nemotron-70B-Reward-HF python reward_server_hf.py

  Then POST to http://localhost:5001/score with JSON:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  Response: {"reward": <float>}
"""

from __future__ import annotations

import os
import sys

try:
    import flask
except ImportError:
    print("Install flask: pip install flask", file=sys.stderr)
    sys.exit(1)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Install torch and transformers: pip install torch transformers", file=sys.stderr)
    sys.exit(1)

MODEL_NAME = os.environ.get("REWARD_MODEL", "nvidia/Llama-3.1-Nemotron-70B-Reward-HF")
HOST = os.environ.get("REWARD_SERVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("REWARD_SERVER_PORT", "5001"))

app = flask.Flask(__name__)
_model = None
_tokenizer = None


def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        print(f"Loading model {MODEL_NAME}...", flush=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("Model loaded.", flush=True)
    return _model, _tokenizer


@app.route("/score", methods=["POST"])
def score():
    """Expect JSON body: {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}. Returns {"reward": float}."""
    data = flask.request.get_json(force=True, silent=True)
    if not data or "messages" not in data:
        return flask.jsonify({"error": "Missing 'messages' in JSON body"}), 400
    messages = data["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return flask.jsonify({"error": "messages must be a list with at least user + assistant"}), 400

    model, tokenizer = get_model_and_tokenizer()
    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
    except Exception as e:
        return flask.jsonify({"error": f"Tokenization failed: {e}"}), 400

    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )

    # Reward = logit of first token (index 0) at first generated position (HF README).
    scores = out.get("scores")
    if not scores or len(scores) == 0:
        return flask.jsonify({"error": "No scores in generate output"}), 500
    first_step_logits = scores[0]  # [batch, vocab_size]
    reward = first_step_logits[0][0].item()
    return flask.jsonify({"reward": reward})


@app.route("/health", methods=["GET"])
def health():
    return flask.jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, threaded=False)
