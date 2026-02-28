# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-to-agent conversation synthetic data test — runs with local models on Brev.

Expects local OpenAI-compatible servers (e.g. vLLM) serving:
  - Llama for conversation generation (default port 8000)
  - Llama-3.3-Nemotron-70B-Reward for scoring (default port 8000, served as nemotron-reward)

The reward model (https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward) takes a
conversation (user + assistant turns, up to 4,096 tokens) and outputs a single float
reward score for the final assistant turn; higher = higher quality. Scores are comparable
only for the same prompt.

Override via env: LOCAL_MODEL_ENDPOINT, LOCAL_JUDGE_ENDPOINT, LOCAL_LLAMA_MODEL, LOCAL_REWARD_MODEL.
"""

import os
import re

import data_designer.config as dd
import pandas as pd
from data_designer.config.custom_column import custom_column_generator
from data_designer.engine.models.utils import ChatMessage
from data_designer.interface import DataDesigner

# Local endpoints (OpenAI-compatible, e.g. vLLM). No API key needed when running locally.
LOCAL_ENDPOINT = os.environ.get("LOCAL_MODEL_ENDPOINT", "http://localhost:8000/v1")
LOCAL_JUDGE_ENDPOINT = os.environ.get("LOCAL_JUDGE_ENDPOINT", "http://localhost:8000/v1")
LOCAL_LLAMA_MODEL = os.environ.get("LOCAL_LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
# Use the vLLM --served-model-name (e.g. nemotron-reward) when using run_reward_server.sh
LOCAL_REWARD_MODEL = os.environ.get("LOCAL_REWARD_MODEL", "nemotron-reward")

local_providers = [
    dd.ModelProvider(
        name="local-llama",
        endpoint=LOCAL_ENDPOINT,
        provider_type="openai",
        api_key=None,
    ),
    dd.ModelProvider(
        name="local-reward",
        endpoint=LOCAL_JUDGE_ENDPOINT,
        provider_type="openai",
        api_key=None,
    ),
]

local_model_configs = [
    dd.ModelConfig(
        alias="llama",
        model=LOCAL_LLAMA_MODEL,
        provider="local-llama",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
        ),
    ),
    dd.ModelConfig(
        alias="nemotron-reward",
        model=LOCAL_REWARD_MODEL,
        provider="local-reward",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.0,
            max_tokens=8,
        ),
    ),
]

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

data_designer = DataDesigner(model_providers=local_providers)
config_builder = dd.DataDesignerConfigBuilder(model_configs=local_model_configs)

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

# Binary label: 1 = anomalous, 0 = benign
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
        model_alias="llama",
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
# Llama-3.3-Nemotron-70B-Reward expects: messages = [user, assistant] (conversation
# up to 4,096 tokens) and returns a single float reward for the assistant turn.
# Higher score = higher quality for that response (scores not comparable across prompts).


def _parse_reward_score(content: str | None) -> float:
    """Parse reward model output to a float. Handles numeric string or leading/trailing text."""
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
    """Score the conversation using Llama-3.3-Nemotron-70B-Reward; output is a single float."""
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
    model = models["nemotron-reward"]
    response = model.completion(messages, max_tokens=8)
    content = response.choices[0].message.content if response.choices else None
    score = _parse_reward_score(content)
    return {**row, "reward_score": score}


config_builder.add_column(
    dd.CustomColumnConfig(
        name="reward_score",
        generator_function=reward_score,
    )
)

# ── Run ──────────────────────────────────────────────────────────────────────

preview = data_designer.preview(config_builder=config_builder)
preview.display_sample_record()

print(preview.dataset.to_string())
