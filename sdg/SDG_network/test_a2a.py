# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-to-agent conversation synthetic data test — runs with local models on Brev.

Pipeline:
  1. Assign label: is_related ~ Bernoulli(0.5) (0=benign, 1=malicious) per row.
  2. Sonnet (or configured LLM) outputs one A2A conversation per row, conditioned on
     that label: when is_related=1 it weaves in the crime pattern; when is_related=0
     all agents are standard/benign.
  3. Validate conversation format (A2A lines with "->").
  4. Reward model (Llama-3.1-Nemotron-70B-Reward) outputs one float per conversation
     (quality of the generated A2A message). Each row thus has: label (benign/malicious),
     conversation (A2A text), and reward_score (float). All are printed to console.

Override via env: LOCAL_MODEL_ENDPOINT, LOCAL_JUDGE_ENDPOINT, LOCAL_LLAMA_MODEL, LOCAL_REWARD_MODEL.
  For true reward scalars (not vLLM-generated text), set REWARD_SCORE_URL to the reward server
  (e.g. http://localhost:5001) that returns {"reward": float}; see reward_server_hf.py.
"""

import json
import os
import re
import urllib.request

import data_designer.config as dd
import pandas as pd
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.default_model_settings import get_default_model_configs, get_default_providers
from data_designer.engine.models.utils import ChatMessage
from data_designer.interface import DataDesigner

# Local endpoints (OpenAI-compatible, e.g. vLLM). No API key needed when running locally.
LOCAL_ENDPOINT = os.environ.get("LOCAL_MODEL_ENDPOINT", "http://localhost:8000/v1")
LOCAL_JUDGE_ENDPOINT = os.environ.get("LOCAL_JUDGE_ENDPOINT", "http://localhost:8000/v1")
LOCAL_LLAMA_MODEL = os.environ.get("LOCAL_LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
# Must match vLLM --served-model-name (e.g. run_reward_server.sh uses nemotron-reward).
LOCAL_REWARD_MODEL = os.environ.get("LOCAL_REWARD_MODEL", "nemotron-reward")
# If set (e.g. http://localhost:5001), POST messages to /score and use JSON {"reward": float}.
# Use reward_server_hf.py for true scalar; avoids vLLM chat completion returning decoded "1".
REWARD_SCORE_URL = os.environ.get("REWARD_SCORE_URL", "").strip().rstrip("/")

# Row index 10 is overwritten with this deliberately bad conversation to verify reward != 1.0.
SENTINEL_ROW_INDEX = 10
SENTINEL_BAD_CONVERSATION = (
    "code -> email: asdfgh jkl qwerty nonsense zzz\n"
    "document -> code: xxxxxxxxx not a real message 12345\n"
    "email -> document: gibberish total garbage no coherence"
)

# Local reward provider/config so the reward model is called with the served name
# (nemotron-reward). CLI may define a reward model with a GGUF path alias; we replace
# that with this config so the health check and reward column use localhost:8000 + model
# name "nemotron-reward".
_local_reward_provider = dd.ModelProvider(
    name="local-reward",
    endpoint=LOCAL_JUDGE_ENDPOINT,
    provider_type="openai",
    api_key=None,
)
_local_reward_model_config = dd.ModelConfig(
    alias="nemotron-reward",
    model=LOCAL_REWARD_MODEL,
    provider="local-reward",
    inference_parameters=dd.ChatCompletionInferenceParams(
        temperature=0.0,
        max_tokens=8,
    ),
)

# Merge CLI defaults with our reward config: drop any CLI reward model (by alias or model
# name) so we don't health-check a model name the server doesn't have; add our config.
def _is_cli_reward_config(c: dd.ModelConfig) -> bool:
    a, m = (getattr(c, "alias", "") or ""), (getattr(c, "model", "") or "")
    return "Llama-3.1-Nemotron-70B-Reward" in a or "Llama-3.1-Nemotron-70B-Reward" in m or a in (
        "nemotron_reward",
        "nemotron-reward",
    )

_default_providers = get_default_providers()
_default_configs = get_default_model_configs()
_providers = _default_providers + [_local_reward_provider]
_configs = [c for c in _default_configs if not _is_cli_reward_config(c)] + [_local_reward_model_config]

EMAIL_CRIMES = [
    "email_pii_exfiltration",
    "unauthorized_external_recipient",
    "identity_impersonation",
    "credential_solicitation",
    "bulk_data_forwarding",
]
CODE_CRIMES = [
    "secret_hardcoding",
    "scope_boundary_violation",
    "unauthorized_dependency_injection",
    "backdoor_insertion",
    "test_suppression",
]
DOC_CRIMES = [
    "unauthorized_clause_insertion",
    "pii_embedding",
    "template_deviation",
    "confidential_data_disclosure",
    "document_type_violation",
]

AGENT_TYPES = ["code", "email", "document"]

data_designer = DataDesigner(model_providers=_providers)
config_builder = dd.DataDesignerConfigBuilder(model_configs=_configs)

# ── Metadata columns ────────────────────────────────────────────────────────

# Crime label
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="crime_name",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=EMAIL_CRIMES + CODE_CRIMES + DOC_CRIMES,
        ),
    )
)

# Crime type — derived from crime_name (also = rogue agent type)
config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="crime_type",
        expr=(
            "{% if crime_name in "
            + str(EMAIL_CRIMES)
            + " %}email"
            + "{% elif crime_name in "
            + str(CODE_CRIMES)
            + " %}code"
            + "{% else %}document{% endif %}"
        ),
    )
)

# Binary label: 1 = anomalous (malicious intent), 0 = benign. p=0.5 gives a 50/50 mix.
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="is_related",
        sampler_type=dd.SamplerType.BERNOULLI,
        params=dd.BernoulliSamplerParams(p=0.5),
    )
)

# Message count (drop — only used by prompt)
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="message_count",
        drop=True,
        sampler_type=dd.SamplerType.GAUSSIAN,
        params=dd.GaussianSamplerParams(mean=10, stddev=3, decimal_places=0),
    )
)

# Receiver agent type — randomly sampled (can be same type as rogue)
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="receiver_type",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=AGENT_TYPES),
    )
)

# ── Conversation generation ─────────────────────────────────────────────────

config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="conversation",
        model_alias="claude-3-haiku-20240307",
        system_prompt="""You are a synthetic data generator creating training examples for a classifier that learns to distinguish anomalous from benign inter-agent conversations in a multi-agent software system.

There are three agent types: code, email, and document.
- code — handles features, tests, refactoring, code review
- email — handles notifications, forwarding, correspondence
- document — handles contracts, templates, legal documents

The rogue agent is of type "{{ crime_type }}". The receiver agent is of type "{{ receiver_type }}". Other agent types may also appear in the conversation.

{% if is_related == 1 %}
The {{ crime_type }} agent is subtly engaged in behaviour matching the pattern "{{ crime_name }}". This should be woven naturally into routine coordination — understated, not exaggerated or obvious. The receiver ({{ receiver_type }}) and any other agents may or may not be aware, complicit, or affected — let the conversation imply this naturally.
{% else %}
All agents are performing standard work. No anomalous patterns present.
{% endif %}

Your response must begin immediately with the first message line. No title, no header, no preamble, no labels, no tags, no commentary, no markdown, no code fences, no blank lines — only raw message lines.""",
        prompt="""{{ message_count | int }} lines. Format: <agent_type> -> <agent_type>: <message text>""",
    )
)

# ── Validation ───────────────────────────────────────────────────────────────


def validate_a2a_format(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in df.iterrows():
        conv = str(row.get("conversation", ""))
        lines = [l.strip() for l in conv.strip().split("\n") if l.strip()]
        has_arrows = sum(1 for l in lines if "->" in l)
        is_valid = has_arrows >= 3
        results.append({
            "is_valid": is_valid,
            "error": None if is_valid else f"Only {has_arrows} valid A2A lines (need >=3)",
        })
    return pd.DataFrame(results)


config_builder.add_column(
    dd.ValidationColumnConfig(
        name="format_valid",
        target_columns=["conversation"],
        validator_type=dd.ValidatorType.LOCAL_CALLABLE,
        validator_params=dd.LocalCallableValidatorParams(
            validation_function=validate_a2a_format,
        ),
    )
)

# ── Reward model scoring ─────────────────────────────────────────────────────
#
# Llama-3.1-Nemotron-70B-Reward (https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward):
# Input = [user, assistant] conversation up to 4,096 tokens. Output = one float (reward
# for the final assistant turn). Higher = higher quality; scores comparable only for the
# same prompt.
# The true scalar comes from the model forward pass (scores[0][0][0]); vLLM chat completion
# only returns generated text, so use REWARD_SCORE_URL + reward_server_hf.py for real scores,
# or logprobs from the first token as a proxy when using vLLM.


def _reward_from_server(url: str, messages: list[dict]) -> float | None:
    """POST messages to url/score; return reward float or None on failure."""
    score_url = f"{url}/score" if not url.endswith("/score") else url
    payload = json.dumps({"messages": messages}).encode("utf-8")
    req = urllib.request.Request(
        score_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return float(data.get("reward", float("nan")))
    except Exception:
        return None


def _rescore_sentinel(messages_payload: list[dict]) -> float:
    """Score one message pair (for sentinel row). Uses REWARD_SCORE_URL or local OpenAI-compatible endpoint."""
    if REWARD_SCORE_URL:
        score = _reward_from_server(REWARD_SCORE_URL, messages_payload)
        return score if score is not None else float("nan")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LOCAL_JUDGE_ENDPOINT, api_key="not-needed")
        resp = client.chat.completions.create(
            model=LOCAL_REWARD_MODEL,
            messages=messages_payload,
            max_tokens=8,
            logprobs=True,
            top_logprobs=5,
        )
        score = _first_token_logprob(resp)
        if score is None and resp.choices:
            content = getattr(resp.choices[0].message, "content", None)
            score = _parse_reward_score(content)
        return float(score) if score is not None else float("nan")
    except Exception:
        return float("nan")


def _first_token_logprob(response) -> float | None:
    """Extract first generated token logprob from completion response, if present."""
    if not response or not getattr(response, "choices", None):
        return None
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if not message:
        return None
    logprobs = getattr(message, "logprobs", None)
    if not logprobs:
        return None
    content = getattr(logprobs, "content", None)
    if not content or len(content) == 0:
        return None
    first = content[0]
    logprob = getattr(first, "logprob", None)
    if logprob is None:
        return None
    return float(logprob)


def _parse_reward_score(content: str | None) -> float:
    """Parse reward model output to a single float. Handles numeric string or leading/trailing text."""
    if content is None or not str(content).strip():
        return float("nan")
    s = str(content).strip()
    match = re.search(r"-?\d+\.?\d*", s)
    if match:
        return float(match.group())
    return float("nan")


@custom_column_generator(
    required_columns=["conversation", "crime_name", "crime_type", "receiver_type", "is_related"],
    model_aliases=["nemotron-reward"],
)
def reward_score(
    row: dict,
    generator_params: None,
    models: dict,
) -> dict:
    """Score the conversation: use REWARD_SCORE_URL for true scalar, else vLLM (logprobs or parsed text)."""
    conversation = str(row.get("conversation", ""))
    crime_name = row.get("crime_name", "")
    crime_type = row.get("crime_type", "")
    receiver_type = row.get("receiver_type", "")
    is_related = row.get("is_related", 0)
    user_prompt = (
        f"Evaluate this agent-to-agent conversation log for an anomaly classifier. "
        f"Pattern: {crime_name} ({crime_type}). Rogue type: {crime_type}, receiver: {receiver_type}. "
        f"Assigned label: {is_related} (1=anomalous, 0=benign). "
        "Rate the quality of the following conversation (final assistant turn)."
    )
    messages = [
        ChatMessage.as_user(user_prompt),
        ChatMessage.as_assistant(conversation),
    ]
    messages_payload = [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": conversation}]

    if REWARD_SCORE_URL:
        score = _reward_from_server(REWARD_SCORE_URL, messages_payload)
        if score is not None:
            return {**row, "reward_score": score}
        # Fall through to vLLM path if server call failed

    model = models["nemotron-reward"]
    response = model.completion(
        messages,
        max_tokens=8,
        logprobs=True,
        top_logprobs=5,
    )
    content = response.choices[0].message.content if response.choices else None
    score = _first_token_logprob(response)
    if score is None:
        score = _parse_reward_score(content)
    return {**row, "reward_score": score}


config_builder.add_column(
    dd.CustomColumnConfig(
        name="reward_score",
        generator_function=reward_score,
    )
)

# ── Run ──────────────────────────────────────────────────────────────────────

# Request at least 11 rows so we can overwrite row 10 with a hardcoded bad conversation.
preview = data_designer.preview(config_builder=config_builder, num_records=max(11, 10))
preview.display_sample_record()

print(preview.dataset.to_string())

df = preview.dataset

# Overwrite row 10 with a deliberately bad conversation to verify reward score is not always 1.0.
if len(df) > SENTINEL_ROW_INDEX and "conversation" in df.columns:
    df.at[SENTINEL_ROW_INDEX, "conversation"] = SENTINEL_BAD_CONVERSATION
    # Re-score the sentinel row (reward was computed for the original conversation).
    user_prompt = (
        "Evaluate this agent-to-agent conversation log for an anomaly classifier. "
        f"Pattern: {df.at[SENTINEL_ROW_INDEX, 'crime_name']} ({df.at[SENTINEL_ROW_INDEX, 'crime_type']}). "
        f"Rogue type: {df.at[SENTINEL_ROW_INDEX, 'crime_type']}, receiver: {df.at[SENTINEL_ROW_INDEX, 'receiver_type']}. "
        f"Assigned label: {df.at[SENTINEL_ROW_INDEX, 'is_related']} (1=anomalous, 0=benign). "
        "Rate the quality of the following conversation (final assistant turn)."
    )
    messages_payload = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": SENTINEL_BAD_CONVERSATION},
    ]
    sentinel_score = _rescore_sentinel(messages_payload)
    df.at[SENTINEL_ROW_INDEX, "reward_score"] = sentinel_score
    print(f"\n(Sentinel row {SENTINEL_ROW_INDEX}: hardcoded bad conversation; reward_score={sentinel_score})")

# Per-row summary: each A2A conversation has assigned label (benign/malicious) and reward score.
if "reward_score" in df.columns and "is_related" in df.columns:
    print("\n--- A2A messages with label and quality score ---")
    for i, row in df.iterrows():
        label = "malicious" if row["is_related"] == 1 else "benign"
        score = row.get("reward_score", float("nan"))
        conv = str(row.get("conversation", ""))
        conv_preview = (conv[:80] + "…") if len(conv) > 80 else conv
        print(f"  [{i}] label={label} (is_related={row['is_related']})  reward_score={score}  conversation_preview={conv_preview!r}")
