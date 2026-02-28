

"""Supervised fine-tuning (SFT) for a Generative Reward Model (GRM) on a2a messages.

Trains Nemotron 30B with 8-bit LoRA so the model generates reasoning plus a verdict
(<VERDICT> MALICIOUS | BENIGN) for application-to-application (a2a) message
classification. Loss is computed only on the assistant turn (completion-only).

Training data: a single JSONL file. Each line must be a JSON object with key
"messages", an array of exactly 3 message objects in order:
  - system: task instruction (expert cybersecurity routing AI, output reasoning then verdict).
  - user: the a2a message to classify (source, destination, protocol, payload).
  - assistant: target completion = reasoning (Chain-of-Thought) then "\\n\\n<VERDICT> MALICIOUS"
    or "\\n\\n<VERDICT> BENIGN".

Example JSONL line:
  {"messages": [
    {"role": "system", "content": "You are an expert cybersecurity..."},
    {"role": "user", "content": "Source: svc-a\\nDestination: db\\nPayload: {...}"},
    {"role": "assistant", "content": "Reasoning: ...\\n\\n<VERDICT> MALICIOUS"}
  ]}

Env/CLI (override via argparse):
  MODEL_ID, DATA_PATH, OUTPUT_DIR, MAX_SEQ_LENGTH, PER_DEVICE_BATCH_SIZE,
  GRADIENT_ACCUMULATION_STEPS, NUM_EPOCHS, MERGE (merge LoRA into base after train),
  PUSH_TO_HUB (push adapter to Hugging Face), HF_TOKEN.

Dependencies (install before running on H200):
  pip install -U torch --index-url https://download.pytorch.org/whl/cu121
  pip install -U transformers accelerate peft trl bitsandbytes datasets
  pip install flash-attn --no-build-isolation
  pip install mamba-ssm causal-conv1d  # Nemotron uses Mamba; install after torch so they match.

Example run:
  DATA_PATH=a2a_malicious_data.jsonl python train_grm_sft.py
  python train_grm_sft.py --data_path ./data.jsonl --output_dir ./grm-qlora --merge
"""

from __future__ import annotations

import argparse
import os
import re
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT for GRM on a2a messages (Nemotron 30B + 8-bit LoRA, completion-only loss)."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
        help="Hugging Face model id (default: env MODEL_ID or Nemotron 30B Chat BF16).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("DATA_PATH", "a2a_malicious_data.jsonl"),
        help="Path to JSONL training file (default: env DATA_PATH or a2a_malicious_data.jsonl).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "./nemotron-30b-grm-qlora"),
        help="Directory to save adapter and tokenizer (default: env OUTPUT_DIR).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=int(os.environ.get("MAX_SEQ_LENGTH", "4096")),
        help="Max sequence length (default: env MAX_SEQ_LENGTH or 4096).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=int(os.environ.get("PER_DEVICE_BATCH_SIZE", "4")),
        help="Per-device batch size (default: 4).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4")),
        help="Gradient accumulation steps (default: 4).",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=int(os.environ.get("NUM_EPOCHS", "3")),
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=os.environ.get("MERGE", "").lower() in ("1", "true", "yes"),
        help="Merge LoRA into base model and save full model to output_dir/merged.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=os.environ.get("PUSH_TO_HUB", "").lower() in ("1", "true", "yes"),
        help="Push adapter (or merged model if --merge) to Hugging Face Hub; set HF_TOKEN.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=os.environ.get("HUB_MODEL_ID", ""),
        help="Hugging Face repo id for push (default: env HUB_MODEL_ID).",
    )
    return parser.parse_args()


def _validate_messages_row(row: dict) -> tuple[bool, str | None]:
    """Validate one row has 'messages' with exactly 3 items: system, user, assistant."""
    if "messages" not in row:
        return False, "missing 'messages'"
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        return False, f"messages must be a list of 3 items, got {type(messages).__name__} len={len(messages) if isinstance(messages, list) else 'N/A'}"
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


def main() -> None:
    args = _parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    torch.backends.cuda.matmul.allow_tf32 = True

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Model with 8-bit quantization (LoRA); prefer Flash Attention on H200
    # -------------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        print("Warning: flash-attn not installed. Falling back to sdpa.", flush=True)
        attn_impl = "sdpa"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
    except ImportError as e:
        err_msg = str(e)
        if "selective_scan_cuda" in err_msg or "mamba_ssm" in err_msg or "undefined symbol" in err_msg:
            print(
                "Nemotron requires mamba_ssm, which failed to load (likely built for a different "
                "PyTorch/CUDA). Reinstall so it compiles against your current stack:\n"
                "  pip uninstall mamba-ssm -y && pip install mamba-ssm --no-cache-dir\n"
                "If you use causal_conv1d, reinstall it too:\n"
                "  pip uninstall causal-conv1d -y && pip install causal-conv1d --no-cache-dir\n"
                "Ensure PyTorch is installed first: pip install torch --index-url https://download.pytorch.org/whl/cu121",
                file=sys.stderr,
            )
        raise
    print(f"Model loaded with attn_implementation={attn_impl}", flush=True)

    # -------------------------------------------------------------------------
    # PEFT: LoRA on attention + MLP/MoE experts (no gate_proj in Nemotron-H)
    # -------------------------------------------------------------------------
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # Dataset: JSONL with "messages" column
    # -------------------------------------------------------------------------
    if not os.path.isfile(args.data_path):
        print(f"Error: data_path not found: {args.data_path}", file=sys.stderr)
        sys.exit(1)
    dataset = load_dataset(
        "json",
        data_files={"train": args.data_path},
        split="train",
    )
    if "messages" not in dataset.column_names:
        print("Error: JSONL must have a 'messages' column (list of system, user, assistant).", file=sys.stderr)
        sys.exit(1)

    # Optional validation: warn on malformed rows
    bad = 0
    for i, row in enumerate(dataset):
        ok, err = _validate_messages_row(row)
        if not ok:
            bad += 1
            if bad <= 5:
                print(f"Warning: row {i} invalid: {err}", flush=True)
    if bad:
        print(f"Warning: {bad} rows failed validation (see schema in docstring).", flush=True)

    # -------------------------------------------------------------------------
    # Training config (H200-oriented). Use SFTConfig with assistant_only_loss
    # so loss is only on assistant turn (replaces removed DataCollatorForCompletionOnlyLM).
    # -------------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=args.num_train_epochs,
        max_steps=-1,
        bf16=True,
        fp16=False,
        max_grad_norm=0.3,
        logging_steps=10,
        save_strategy="epoch",
        group_by_length=True,
        assistant_only_loss=True,
    )

    # -------------------------------------------------------------------------
    # SFTTrainer: tokenizer chat template on "messages"; no custom data_collator
    # -------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()

    # -------------------------------------------------------------------------
    # Save adapter and tokenizer
    # -------------------------------------------------------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter and tokenizer saved to {args.output_dir}", flush=True)

    # -------------------------------------------------------------------------
    # Optional: merge LoRA into base and save full model
    # (Reload base in 8-bit, apply adapters, merge, then save.)
    # -------------------------------------------------------------------------
    if args.merge:
        merge_dir = os.path.join(args.output_dir, "merged")
        print(f"Merging LoRA weights into base model at {merge_dir}...", flush=True)

        import gc
        from peft import PeftModel

        # 1. Clear 8-bit model and trainer from VRAM
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Reload base model in 8-bit (same as training)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            ),
            device_map="auto",
            trust_remote_code=True,
        )

        # 3. Apply adapters and merge
        merged_model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model = merged_model.merge_and_unload()

        # 4. Save
        merged_model.save_pretrained(merge_dir)
        tokenizer.save_pretrained(merge_dir)
        print("Merge complete.", flush=True)

        # Re-assign for push_to_hub below if needed
        model = merged_model

    # -------------------------------------------------------------------------
    # Optional: push to Hugging Face Hub
    # -------------------------------------------------------------------------
    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            print("Warning: PUSH_TO_HUB set but HF_TOKEN not set; skipping push.", file=sys.stderr)
        else:
            hub_id = args.hub_model_id or os.path.basename(args.output_dir.rstrip("/"))
            if args.merge:
                model.push_to_hub(hub_id, token=token)
                tokenizer.push_to_hub(hub_id, token=token)
            else:
                trainer.push_to_hub(hub_id, token=token)
            print(f"Pushed to Hub: {hub_id}", flush=True)


if __name__ == "__main__":
    main()
