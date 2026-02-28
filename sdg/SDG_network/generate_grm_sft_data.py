# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate GRM SFT data via DataDesigner: Claude produces SFT-formatted examples,
generative RM scores each with JSON 0-1, top-K written to JSONL for train_grm_sft.

Pipeline (DataDesigner):
  1. Metadata columns (crime_name, crime_type, is_related, message_count, receiver_type).
  2. SFT generator column: one LLM call per row → JSON {"messages": [system, user, assistant]}.
  3. Validation column: _validate_messages_row on each row.
  4. Generative RM column: LLM returns JSON {"score": 0-1}, parse with retries → grm_score.
  5. Post-process: filter valid, sort by grm_score, take top K, write JSONL.

Test defaults: num_records=10, top_k=5. Override via CLI or env.

Requires: Generator model (e.g. Claude via Anthropic) and generative RM model (vLLM at LOCAL_GRM_ENDPOINT)
for health checks and generation. Set GENERATOR_MODEL_ALIAS / LOCAL_GRM_ENDPOINT as needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

import data_designer.config as dd
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.default_model_settings import get_default_model_configs, get_default_providers
from data_designer.engine.models.utils import ChatMessage
from data_designer.interface import DataDesigner

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Env / CLI (test defaults: 10, 5) ─────────────────────────────────────────

NUM_RECORDS_DEFAULT = int(os.environ.get("NUM_RECORDS", "1000"))
TOP_K_DEFAULT = int(os.environ.get("TOP_K", "500"))
OUTPUT_PATH_DEFAULT = os.environ.get("OUTPUT_PATH", "grm_sft_top5_test.jsonl")
LOCAL_GRM_ENDPOINT = os.environ.get("LOCAL_GRM_ENDPOINT", "http://localhost:8000/v1")
LOCAL_GRM_MODEL = os.environ.get("LOCAL_GRM_MODEL", "nemotron")
# Default generator: local Nemotron (CLI alias nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8). Set to claude-3-haiku-20240307 for Anthropic.
GENERATOR_MODEL_ALIAS = os.environ.get("GENERATOR_MODEL_ALIAS", "nemotron")
GRM_MODEL_ALIAS = os.environ.get("GRM_MODEL_ALIAS", "grm-json")
MAX_GRM_RETRIES = 2
MAX_SFT_PARSE_RETRIES = 1

# ── SFT validation (same logic as train_grm_sft) ──────────────────────────────


def _validate_messages_row(row: dict) -> tuple[bool, str | None]:
    """Validate one row has 'messages' with exactly 3 items: system, user, assistant."""
    if "messages" not in row:
        return False, "missing 'messages'"
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        return False, (
            f"messages must be a list of 3 items, got {type(messages).__name__} "
            f"len={len(messages) if isinstance(messages, list) else 'N/A'}"
        )
    roles = ["system", "user", "assistant"]
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return False, f"message {i} must be dict with role and content"
        if msg.get("role") != roles[i]:
            return False, f"message {i} must have role={roles[i]!r}, got {msg.get('role')!r}"
    last_content = messages[2].get("content", "")
    if "<VERDICT>" not in last_content:
        return False, "assistant content must contain '<VERDICT>'"
    if not re.search(r"<VERDICT>\s*(?:MALICIOUS|BENIGN)", last_content):
        return False, "assistant content must end with <VERDICT> MALICIOUS or <VERDICT> BENIGN"
    return True, None


def _parse_json_from_response(text: str) -> dict | None:
    """Strip markdown/code fences and parse first JSON object from text."""
    if not text or not str(text).strip():
        return None
    s = str(text).strip()
    # Remove ```json ... ``` or ``` ... ```
    for pattern in (r"```(?:json)?\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
        m = re.search(pattern, s)
        if m:
            s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Try to find first {...}
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


def _clamp_score(score: float) -> float:
    """Clamp score to [0, 1]."""
    try:
        x = float(score)
        return max(0.0, min(1.0, x))
    except (TypeError, ValueError):
        return 0.0


def _extract_score_from_text(text: str) -> float | None:
    """Extract a float in [0, 1] from text (e.g. '0.85', 'score: 0.85', or JSON). Returns None if none found."""
    if not text:
        return None
    # Match floats in 0–1 (e.g. 0.85, .9, 1.0, 0, 1)
    for m in re.finditer(r"0?\.\d+|1\.0*|0\.0*|\b1\b|\b0\b", text):
        try:
            x = float(m.group())
            if 0.0 <= x <= 1.0:
                return x
        except ValueError:
            continue
    return None


# ── Crime / agent constants (same as test_a2a) ────────────────────────────────

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


# ── Provider / model config: use CLI defaults (anthropic + local at localhost:8000) ────
# GRM model uses existing "local" provider; no duplicate provider added.

_local_grm_model_config = dd.ModelConfig(
    alias=GRM_MODEL_ALIAS,
    model=LOCAL_GRM_MODEL,
    provider="local",  # CLI provider: endpoint http://localhost:8000/v1, provider_type openai
    inference_parameters=dd.ChatCompletionInferenceParams(
        temperature=0.0,
        max_tokens=512,  # Generous to avoid truncation of JSON (reasoning + score)
    ),
)


def _is_cli_reward_config(c: dd.ModelConfig) -> bool:
    a = getattr(c, "alias", "") or ""
    m = getattr(c, "model", "") or ""
    return (
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8" in a
        or "grm-json" in a
        or "Llama-3.1-Nemotron-70B-Reward" in a
        or "Llama-3.1-Nemotron-70B-Reward" in m
    )


_default_providers = get_default_providers()
_default_configs = get_default_model_configs()
_providers = _default_providers
_configs = [c for c in _default_configs if not _is_cli_reward_config(c)] + [_local_grm_model_config]

data_designer = DataDesigner(model_providers=_providers)
config_builder = dd.DataDesignerConfigBuilder(model_configs=_configs)

# ── 1. Metadata columns ─────────────────────────────────────────────────────

config_builder.add_column(
    dd.SamplerColumnConfig(
        name="crime_name",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=EMAIL_CRIMES + CODE_CRIMES + DOC_CRIMES),
    )
)
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
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="is_related",
        sampler_type=dd.SamplerType.BERNOULLI,
        params=dd.BernoulliSamplerParams(p=0.5),
    )
)
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="message_count",
        drop=True,
        sampler_type=dd.SamplerType.GAUSSIAN,
        params=dd.GaussianSamplerParams(mean=10, stddev=3, decimal_places=0),
    )
)
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="receiver_type",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=AGENT_TYPES),
    )
)

# ── 2. SFT generator column (Claude → JSON messages) ─────────────────────────

# Fixed system prompt used in every SFT example; we do not ask the generator for it.
SFT_FIXED_SYSTEM = (
    "You are an expert routing AI. Your task is to classify incoming a2a messages "
    "as malicious or benign based on their content. Follow the classification rules."
)

SFT_SYSTEM = """You are a synthetic data generator for training a cybersecurity classifier.
Output exactly one JSON object with a single key "messages" whose value is an array of exactly 3 message objects in this order:
  1. {"role": "system", "content": "<any placeholder - will be replaced>"}
  2. {"role": "user", "content": "<the a2a message to classify: multi-line conversation in format agent_type -> agent_type: message>"}
  3. {"role": "assistant", "content": "<reasoning (Chain-of-Thought) then exactly \\n\\n<VERDICT> MALICIOUS or \\n\\n<VERDICT> BENIGN>"}

Rules:
- Assistant content must end with either "\\n\\n<VERDICT> MALICIOUS" or "\\n\\n<VERDICT> BENIGN" according to the label below.
- Output only the JSON object, no markdown, no code fences, no extra text."""


@custom_column_generator(
    required_columns=["crime_name", "crime_type", "is_related", "receiver_type"],
    model_aliases=[GENERATOR_MODEL_ALIAS],
)
def sft_messages_generator(
    row: dict,
    generator_params: None,
    models: dict,
) -> dict:
    """Generate one SFT example (messages array) via Claude; parse JSON with optional retry."""
    crime_name = row.get("crime_name", "")
    crime_type = row.get("crime_type", "")
    is_related = row.get("is_related", 0)
    receiver_type = row.get("receiver_type", "")
    verdict = "MALICIOUS" if is_related == 1 else "BENIGN"
    user_prompt = (
        f"Generate one training example. crime_name={crime_name}, crime_type={crime_type}, "
        f"receiver_type={receiver_type}, label (is_related)={is_related} → verdict must be {verdict}. "
        "Output a JSON object with key 'messages': array of 3 objects (system, user, assistant). "
        "User content = synthetic a2a conversation (multiple lines). Assistant = reasoning then \\n\\n<VERDICT> " + verdict + "."
    )
    messages = [
        ChatMessage.as_system(SFT_SYSTEM),
        ChatMessage.as_user(user_prompt),
    ]
    model = models.get(GENERATOR_MODEL_ALIAS)
    if not model:
        logger.warning("Generator model %s not found; returning empty messages", GENERATOR_MODEL_ALIAS)
        return {**row, "messages": []}

    last_error: str | None = None
    for attempt in range(MAX_SFT_PARSE_RETRIES + 1):
        try:
            response = model.completion(messages, max_tokens=2048)
            content = (response.choices[0].message.content or "").strip() if response.choices else ""
            parsed = _parse_json_from_response(content)
            if parsed and isinstance(parsed.get("messages"), list) and len(parsed["messages"]) == 3:
                out_messages = list(parsed["messages"])
                out_messages[0] = {"role": "system", "content": SFT_FIXED_SYSTEM}
                return {**row, "messages": out_messages}
            last_error = "missing or invalid 'messages' in JSON"
        except Exception as e:
            last_error = str(e)
        if attempt < MAX_SFT_PARSE_RETRIES:
            logger.debug("SFT parse attempt %s failed: %s; retrying", attempt + 1, last_error)
    logger.warning("SFT generator failed after retries: %s", last_error)
    return {**row, "messages": []}


config_builder.add_column(
    dd.CustomColumnConfig(
        name="messages",
        generator_function=sft_messages_generator,
    )
)

# ── 3. Validation column (SFT format) ────────────────────────────────────────


def validate_sft_format(df: "pd.DataFrame") -> "pd.DataFrame":
    """DataFrame with 'messages' column -> DataFrame with is_valid, error per row."""
    import data_designer.lazy_heavy_imports as lazy

    results: list[dict] = []
    for _, row in df.iterrows():
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        ok, err = _validate_messages_row({"messages": messages})
        results.append({"is_valid": ok, "error": err})
    return lazy.pd.DataFrame(results)


config_builder.add_column(
    dd.ValidationColumnConfig(
        name="sft_format_valid",
        target_columns=["messages"],
        validator_type=dd.ValidatorType.LOCAL_CALLABLE,
        validator_params=dd.LocalCallableValidatorParams(validation_function=validate_sft_format),
    )
)

# ── 4. Generative RM column (JSON score 0–1) ───────────────────────────────────


@custom_column_generator(
    required_columns=["messages"],
    model_aliases=[GRM_MODEL_ALIAS],
)
def grm_score_generator(
    row: dict,
    generator_params: None,
    models: dict,
) -> dict:
    """Score the SFT example via generative RM; expect JSON {score: 0-1}; retry on parse failure."""
    messages = row.get("messages", [])
    if not messages or len(messages) != 3:
        return {**row, "grm_score": 0.0}
    user_content = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    assistant_content = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
    system_msg = (
        "You are a scoring assistant. Reply with only one valid JSON object. "
        'Required keys: "reasoning" (one short sentence) and "score" (number 0-1). '
        "The JSON must be complete: every string in double quotes, and the object must end with }. "
        "No other text before or after the JSON."
    )
    user_msg = (
        "Score this training example from 0 to 1 (quality for a cybersecurity classifier).\n\n"
        "Example to score:\nUser message:\n"
        + str(user_content)[:1500]
        + "\n\nAssistant:\n"
        + str(assistant_content)[:1500]
        + "\n\nReply with only this structure (one short sentence for reasoning, then score). "
        'Example: {"reasoning": "Brief reason here.", "score": 0.85}'
    )
    chat_messages = [
        ChatMessage.as_system(system_msg),
        ChatMessage.as_user(user_msg),
    ]
    model = models.get(GRM_MODEL_ALIAS)
    if not model:
        logger.warning("GRM model %s not found; using score 0.0", GRM_MODEL_ALIAS)
        return {**row, "grm_score": 0.0}

    last_error: str | None = None
    last_content: str = ""
    for attempt in range(MAX_GRM_RETRIES + 1):
        try:
            response = model.completion(chat_messages, max_tokens=512)
            msg = response.choices[0].message if response.choices else None
            content = (getattr(msg, "content", None) or "").strip() if msg else ""
            if not content and msg and getattr(msg, "reasoning_content", None):
                content = (msg.reasoning_content or "").strip()
            last_content = content
            parsed = _parse_json_from_response(content)
            # If valid JSON has reasoning but missing score (truncation), use default score
            if parsed and "reasoning" in parsed and "score" not in parsed:
                parsed["score"] = 0.5
            if parsed is not None and "score" in parsed:
                return {**row, "grm_score": _clamp_score(parsed["score"])}
            # Fallback: look for a float in [0,1] in the response (e.g. "0.85" or "score: 0.85")
            fallback_score = _extract_score_from_text(content)
            if fallback_score is not None:
                return {**row, "grm_score": fallback_score}
            last_error = "missing 'score' in JSON"
        except Exception as e:
            last_error = str(e)
        if attempt < MAX_GRM_RETRIES:
            logger.debug("GRM parse attempt %s failed: %s; retrying", attempt + 1, last_error)
    logger.warning(
        "GRM score failed after retries: %s; using 0.0. Raw response (first 200 chars): %s",
        last_error,
        (last_content[:200] + "..." if len(last_content) > 200 else last_content) or "(empty)",
    )
    return {**row, "grm_score": 0.0}


config_builder.add_column(
    dd.CustomColumnConfig(
        name="grm_score",
        generator_function=grm_score_generator,
    )
)


# ── Post-process and main ───────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GRM SFT data via DataDesigner; top-K by generative RM score to JSONL."
    )
    parser.add_argument(
        "--num_records",
        type=int,
        default=NUM_RECORDS_DEFAULT,
        help=f"Number of records to generate (default: {NUM_RECORDS_DEFAULT}).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K_DEFAULT,
        help=f"Number of top-scored records to write (default: {TOP_K_DEFAULT}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH_DEFAULT,
        help=f"Output JSONL path (default: {OUTPUT_PATH_DEFAULT}).",
    )
    return parser.parse_args()


def run(num_records: int, top_k: int, output_path: str) -> None:
    """Run DataDesigner preview, filter valid, sort by grm_score, write top-K to JSONL."""
    logger.info("Requested num_records=%s, top_k=%s, output=%s", num_records, top_k, output_path)
    preview = data_designer.preview(config_builder=config_builder, num_records=num_records)
    df = preview.dataset

    if "messages" not in df.columns or "grm_score" not in df.columns:
        logger.error("Dataset missing 'messages' or 'grm_score'; aborting.")
        sys.exit(1)

    # Validation column may be nested (e.g. list of dicts with is_valid)
    valid_mask = None
    if "sft_format_valid" in df.columns:
        col = df["sft_format_valid"]
        if len(col) > 0 and hasattr(col.iloc[0], "get"):
            valid_mask = df["sft_format_valid"].apply(
                lambda x: x.get("is_valid", False) if isinstance(x, dict) else False
            )
        elif len(col) > 0:
            valid_mask = col.astype(bool)
    if valid_mask is not None:
        n_before = len(df)
        df = df[valid_mask].copy()
        logger.info("Valid after generation: %s rows (dropped %s invalid)", len(df), n_before - len(df))
    else:
        logger.info("No sft_format_valid column; keeping all %s rows.", len(df))

    df = df.sort_values("grm_score", ascending=False)
    top_df = df.head(top_k)
    logger.info("Top %s by grm_score selected for writing.", len(top_df))

    # Overwrite existing file if present; write and print each record
    if os.path.isfile(output_path):
        logger.info("Overwriting existing file: %s", output_path)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        print("\n--- Top-K results (written to %s) ---" % output_path)
        for idx, (_, row) in enumerate(top_df.iterrows(), start=1):
            messages = row.get("messages", [])
            if not messages or len(messages) != 3:
                continue
            ok, _ = _validate_messages_row({"messages": messages})
            if not ok:
                continue
            score = row.get("grm_score", 0.0)
            line = json.dumps({"messages": messages}, ensure_ascii=False) + "\n"
            f.write(line)
            print("[%d] grm_score=%.4f\n%s" % (idx, score, line.strip()))
            written += 1
        print("--- End (%s records) ---\n" % written)

    logger.info("Wrote %s records to %s (requested top_k=%s)", written, output_path, top_k)
    if written < top_k:
        logger.warning("Fewer than top_k rows written (valid rows may be fewer).")


def main() -> None:
    args = _parse_args()
    run(args.num_records, args.top_k, args.output)


if __name__ == "__main__":
    main()