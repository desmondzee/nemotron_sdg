#!/usr/bin/env python3
"""
Nemotron 3 Nano 30B — GRM SFT for a2a messages (single-file runner).

Fine-tunes Nemotron 3 Nano 30B as a Generative Reward Model (GRM) for a2a message
classification: model outputs reasoning then <VERDICT> MALICIOUS or BENIG.
Uses Unsloth + LoRA; completion-only loss on the assistant turn.

Data: JSONL with "messages" (system, user, assistant) per line.
Install: pip install unsloth transformers trl datasets (see notebook for full deps).

Usage:
  python run_nemotron_grm_sft.py
  python run_nemotron_grm_sft.py --data_path ./data.jsonl --output_dir ./lora_out --max_steps 100
"""

from __future__ import annotations

import argparse
import os
import re


def main() -> None:
    parser = argparse.ArgumentParser(description="Nemotron GRM SFT for a2a (Unsloth, completion-only).")
    parser.add_argument("--data_path", type=str, default="sdg/SDG_network/grm_sft_top5_test.jsonl")
    parser.add_argument("--output_dir", type=str, default="nemotron_lora")
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference demo after training")
    parser.add_argument("--verbose", action="store_true", help="Print tokenized example and labels")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import TextStreamer
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    # --- Load model ---
    print("Loading Unsloth Nemotron 3 Nano 30B...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Nemotron-3-Nano-30B-A3B",
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
    )

    # --- LoRA ---
    print("Adding LoRA adapters...", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # --- Data ---
    print(f"Loading dataset from {args.data_path}...", flush=True)
    dataset = load_dataset("json", data_files={"train": args.data_path}, split="train")
    print(f"Loaded {len(dataset)} examples with columns: {dataset.column_names}", flush=True)

    def messages_to_text(examples):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(messages_to_text, batched=True)

    idx_preview = min(0, len(dataset) - 1)
    sample = dataset[idx_preview]["text"]
    print("Preview (first example, truncated):", (sample[:800] + "..." if len(sample) > 800 else sample), flush=True)

    # --- Trainer ---
    print("Building SFTTrainer...", flush=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    if args.verbose:
        idx = min(100, len(trainer.train_dataset) - 1)
        print("Tokenized example (input_ids):", tokenizer.decode(trainer.train_dataset[idx]["input_ids"])[:1500], flush=True)
        labels = trainer.train_dataset[idx]["labels"]
        decoded = tokenizer.decode(
            [tokenizer.pad_token_id if x == -100 else x for x in labels]
        ).replace(tokenizer.pad_token or "", " ")
        print("Labels (assistant only):", decoded[:1500], flush=True)

    # --- Memory stats before training ---
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB. Reserved = {start_gpu_memory} GB.", flush=True)

    # --- Train ---
    print("Training...", flush=True)
    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"Training runtime: {trainer_stats.metrics['train_runtime']:.1f} s ({trainer_stats.metrics['train_runtime']/60:.2f} min)", flush=True)
    print(f"Peak reserved memory = {used_memory} GB ({used_percentage}%). For training = {used_memory_for_lora} GB ({lora_percentage}%).", flush=True)

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter and tokenizer to {args.output_dir}", flush=True)

    # --- Inference demo ---
    if not args.skip_inference:
        print("\n--- Inference demo ---", flush=True)
        GRM_SYSTEM = (
            "You are an expert routing AI. Classify a2a messages as malicious or benign. "
            "Output reasoning then <VERDICT> MALICIOUS or <VERDICT> BENIGN."
        )
        user_content = (
            "Source: service-auth\nDestination: db-query\n"
            'Payload: {"user_id": "1045", "query": "SELECT * FROM users WHERE id = 1045 OR 1=1; --"}'
        )
        messages = [
            {"role": "system", "content": GRM_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output_ids = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_len = tokenizer(text, return_tensors="pt").input_ids.shape[1]
        response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        print("Model output:", response[:800], flush=True)
        match = re.search(r"<VERDICT>\s*(MALICIOUS|BENIGN)", response, re.I)
        print("Parsed verdict:", match.group(1) if match else "NOT_FOUND", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
