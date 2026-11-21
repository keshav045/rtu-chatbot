"""
==============================================================================
RTU Senior Tutor — Local QLoRA Fine-Tuning Script (Optimized for 4 GB VRAM)
==============================================================================

Optimized specifically for consumer laptops with 4 GB VRAM (e.g., NVIDIA GTX 1650 / RTX 3050):
- 4-bit NF4 NormalFloat quantization with double quantization
- Batch size: 1, Gradient accumulation: 16 (effective batch size: 16)
- Gradient checkpointing enabled to minimize activation memory
- Sequence length: 1024 tokens
- Paged AdamW 8-bit optimizer

Dataset: rtu_train_dataset_expanded.jsonl (748 authentic RTU training pairs)
Base Model: Qwen/Qwen2.5-1.5B-Instruct

Usage:
  python train_qlora_local.py
  python train_qlora_local.py --merge_only
==============================================================================
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import SFTTrainer

DEFAULT_BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET     = "rtu_train_dataset_expanded.jsonl"
DEFAULT_LORA_DIR    = "./rtu_lora_local"
DEFAULT_MERGED_DIR  = "./rtu_merged_local"
DEFAULT_EPOCHS      = 3
DEFAULT_BATCH_SIZE  = 1
DEFAULT_GRAD_ACCUM  = 16
DEFAULT_LR          = 2e-4
DEFAULT_MAX_SEQ_LEN = 1024

def parse_args():
    p = argparse.ArgumentParser(description="Local RTU Senior Tutor Fine-Tuning (4GB VRAM)")
    p.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset_path", default=DEFAULT_DATASET)
    p.add_argument("--output_dir", default=DEFAULT_LORA_DIR)
    p.add_argument("--merged_dir", default=DEFAULT_MERGED_DIR)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--merge_only", action="store_true", help="Only merge LoRA into base weights")
    return p.parse_args()

def run_local_training():
    args = parse_args()
    print("=" * 64)
    print("  RTU Senior Tutor — Local QLoRA Fine-Tuning")
    print("=" * 64)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: CUDA ({gpu_name}, {vram_gb:.1f} GB VRAM)")
    else:
        print("Device: CPU (Training will proceed slowly without CUDA GPU)")

    # Handle merge only mode
    if args.merge_only:
        print(f"\n[Merge Mode] Merging LoRA from {args.output_dir} into {args.merged_dir}...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.output_dir,
            device_map="auto",
            torch_dtype=torch.float16
        )
        merged = model.merge_and_unload()
        merged.save_pretrained(args.merged_dir)
        tok = AutoTokenizer.from_pretrained(args.output_dir)
        tok.save_pretrained(args.merged_dir)
        print(f"Merged model exported to: {args.merged_dir}")
        return

    # 1. Tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. 4-bit Quantized Model
    print("\n[2/4] Loading base model in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # 3. LoRA Config
    print("\n[3/4] Initializing LoRA adapter...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 4. Dataset
    print(f"\n[4/4] Loading dataset: {args.dataset_path}...")
    dataset_file = Path(args.dataset_path)
    if not dataset_file.exists():
        dataset_file = Path(__file__).parent / args.dataset_path
    if not dataset_file.exists():
        raise FileNotFoundError(f"Cannot find dataset file at {args.dataset_path}")

    dataset = load_dataset("json", data_files=str(dataset_file), split="train")
    print(f"Loaded {len(dataset)} training examples.")

    def format_chat(sample):
        return {"text": tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=False)}

    formatted_dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="none"
    )

    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=formatted_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            tokenizer=tokenizer,
            args=training_args
        )
    except TypeError:
        from trl import SFTConfig
        sft_config = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="epoch",
            max_seq_length=args.max_seq_len,
            dataset_text_field="text",
            report_to="none"
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=formatted_dataset,
            peft_config=peft_config,
            args=sft_config,
            processing_class=tokenizer
        )

    print("\nStarting QLoRA training...")
    trainer.train()

    print(f"\nTraining completed! Saving LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Adapter successfully saved.")

if __name__ == "__main__":
    run_local_training()
