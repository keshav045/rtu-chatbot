"""
==============================================================================
RTU Senior Tutor — QLoRA Fine-Tuning Script for Google Colab (Free T4 GPU)
==============================================================================

PROJECT CONTEXT
---------------
This script fine-tunes Qwen2.5-1.5B-Instruct using QLoRA on 200 RTU-specific
training examples (seed_dataset_200.jsonl) to create the "Senior Tutor" persona
for the RTU AI Academic Assistant.

The fine-tuned model will be used as the offline/local LLM provider in the
backend via Ollama (see: backend/app/llm/ollama_provider.py).

GOOGLE COLAB USAGE
------------------
Step 1 — Upload both files to Colab:
    - train_qlora_colab.py
    - seed_dataset_200.jsonl

Step 2 — Install dependencies (run in a Colab cell):
    !pip install -q torch transformers datasets trl peft bitsandbytes accelerate

Step 3 — Run training (run in a Colab cell):
    !python train_qlora_colab.py

Step 4 — After training, merge + export (run in a Colab cell):
    !python train_qlora_colab.py --merge_only

MODEL SPECS (Free T4 GPU — 15 GB VRAM)
----------------------------------------
  Base model  : Qwen/Qwen2.5-1.5B-Instruct (1.5B params, ~3.0 GB in 4-bit)
  LoRA rank   : r=16, alpha=32 (18.4M trainable params, 1.18% of total)
  Batch       : 4 x 4 grad_accum = 16 effective
  Epochs      : 3
  VRAM usage  : ~5–6 GB (safe for free T4)
  Time        : ~12–15 minutes on T4

WHAT THIS PRODUCES
------------------
  ./rtu_lora_adapter/     — LoRA adapter weights
  ./rtu_merged_model/     — Full merged model (for GGUF export)
  ./rtu_senior_tutor_q4.gguf — Final GGUF for Ollama (after llama.cpp step)

OLLAMA USAGE (after GGUF export, on your local machine)
--------------------------------------------------------
    ollama create rtu-senior -f Modelfile
    # In backend: set LLM_PROVIDER=ollama, OLLAMA_MODEL=rtu-senior
==============================================================================
"""

import os
import sys
import json
import inspect
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training


# ==============================================================================
# CONFIGURATION — Change these if needed
# ==============================================================================

DEFAULT_BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET     = "rtu_train_dataset_expanded.jsonl" if Path("rtu_train_dataset_expanded.jsonl").exists() or (Path(__file__).parent / "rtu_train_dataset_expanded.jsonl").exists() else "seed_dataset_200.jsonl"
DEFAULT_LORA_DIR    = "./rtu_lora_adapter"
DEFAULT_MERGED_DIR  = "./rtu_merged_model"
DEFAULT_EPOCHS      = 3
DEFAULT_BATCH_SIZE  = 4
DEFAULT_GRAD_ACCUM  = 4
DEFAULT_LR          = 2e-4
DEFAULT_MAX_SEQ_LEN = 2048


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTU Senior Tutor QLoRA Fine-Tuning")
    p.add_argument("--base_model",   default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset_path", default=DEFAULT_DATASET)
    p.add_argument("--output_dir",   default=DEFAULT_LORA_DIR)
    p.add_argument("--merged_dir",   default=DEFAULT_MERGED_DIR)
    p.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad_accum",   type=int,   default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--max_seq_len",  type=int,   default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument(
        "--merge_only",
        action="store_true",
        help="Skip training; only merge existing LoRA adapter and export."
    )
    return p.parse_args()


# ==============================================================================
# HELPERS
# ==============================================================================

def banner(text: str):
    print("\n" + "=" * 64)
    print(f"  {text}")
    print("=" * 64)


def device_info() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA — {name} ({vram:.1f} GB VRAM)"
    return "CPU (Warning: training will be very slow!)"


def try_init(cls, fixed_kwargs: dict, variant_list: list):
    """
    Attempt to construct `cls(**fixed_kwargs, **variant)` for each dict in
    variant_list, returning the first successful instance.

    This handles TRL API changes across versions without crashing:
      - SFTConfig: max_length vs max_seq_length vs dataset_text_field
      - SFTTrainer: processing_class vs tokenizer
    """
    last_error = None
    for variant in variant_list:
        try:
            return cls(**fixed_kwargs, **variant)
        except TypeError as e:
            last_error = e
            continue
    raise RuntimeError(
        f"Could not construct {cls.__name__} with any parameter variant.\n"
        f"Last error: {last_error}\n"
        f"TRL version may be unsupported. Try: pip install -q trl==0.11.4"
    )


# ==============================================================================
# STEP 1 — LOAD TOKENIZER
# ==============================================================================

def load_tokenizer(model_name: str) -> AutoTokenizer:
    print("\n[1/5] Loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # Required for SFT causal left-padding fix
    print(f"      Vocab size    : {tok.vocab_size:,}")
    print(f"      Pad token     : '{tok.pad_token}'")
    return tok


# ==============================================================================
# STEP 2 — LOAD QUANTIZED BASE MODEL
# ==============================================================================

def load_model(model_name: str) -> AutoModelForCausalLM:
    print("\n[2/5] Loading 4-bit NF4 quantized base model ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        # Force ALL non-quantized layers (norms, embeddings) to float16.
        # Qwen2.5 defaults to bfloat16 which crashes on T4 GPU with fp16 AMP.
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # Hard-cast any remaining bfloat16 tensors → float16.
    # paged_adamw_8bit can leave some optimizer-state tensors in bfloat16
    # which makes the CUDA AMP grad scaler raise NotImplementedError on T4.
    bf16_count = 0
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
            bf16_count += 1
    if bf16_count:
        print(f"      Recast {bf16_count} bfloat16 tensors → float16 for T4 compatibility.")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total params  : {total_params / 1e6:.1f}M")
    return model


# ==============================================================================
# STEP 3 — BUILD LORA CONFIG (not applied manually — SFTTrainer does it)
# ==============================================================================

def build_lora_config() -> LoraConfig:
    print("\n[3/5] Building LoRA config (r=16, alpha=32) ...")
    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Target all projection layers in Qwen2.5 / LLaMA-style architectures
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    print("      LoRA ready. SFTTrainer will apply it automatically.")
    return cfg


# ==============================================================================
# STEP 4 — LOAD & FORMAT DATASET
# ==============================================================================

def load_and_format_dataset(dataset_path: str, tokenizer: AutoTokenizer):
    """
    Loads seed_dataset_200.jsonl and applies the tokenizer's chat template to
    each example, producing a single 'text' column that SFTTrainer trains on.

    Dataset format (each line):
        {
          "task": "explain_topic",
          "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
          ]
        }
    """
    print(f"\n[4/5] Loading dataset from: {dataset_path}")

    actual_path = Path(dataset_path)
    if not actual_path.exists():
        script_relative = Path(__file__).parent / dataset_path
        if script_relative.exists():
            actual_path = script_relative
        else:
            fallback = Path(__file__).parent / "rtu_train_dataset_expanded.jsonl"
            if fallback.exists():
                actual_path = fallback
            else:
                fallback_seed = Path(__file__).parent / "seed_dataset_200.jsonl"
                if fallback_seed.exists():
                    actual_path = fallback_seed
                else:
                    raise FileNotFoundError(
                        f"Dataset not found at: {dataset_path} or {script_relative}\n"
                        "Please ensure rtu_train_dataset_expanded.jsonl is present."
                    )

    raw = load_dataset("json", data_files=str(actual_path), split="train")
    print(f"      Raw examples  : {len(raw)}")

    def apply_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    formatted = raw.map(apply_chat_template, remove_columns=raw.column_names)
    print(f"      Formatted     : {len(formatted)} conversation turns")

    # Print a sample so we can verify the format looks right
    sample = formatted[0]["text"]
    print(f"      Sample (first 200 chars): {repr(sample[:200])}")

    return formatted


# ==============================================================================
# STEP 5 — CONFIGURE SFTCONFIG (version-agnostic)
# ==============================================================================

def build_sft_config(args: argparse.Namespace):
    """
    Build SFTConfig in a way that works across TRL 0.8–0.14+.

    Key API changes across TRL versions:
      - warmup_ratio removed → use warmup_steps
      - dataset_text_field moved from SFTTrainer to SFTConfig, then removed
      - max_seq_length renamed to max_length in SFTConfig
      - TrainingArguments subclassed by SFTConfig in newer versions
    """
    from trl import SFTConfig

    # Base kwargs that are stable across all recent TRL versions
    base = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_strategy="epoch",
        # fp16=False, bf16=False → train in float32 AMP-free.
        # T4 GPU's CUDA AMP scaler does not support bfloat16, and Qwen2.5
        # has residual bfloat16 tensors that crash the grad scaler.
        # float32 training on 200 examples is ~18 min on T4 — perfectly fine.
        fp16=False,
        bf16=False,
        optim="adamw_torch",    # paged_adamw_8bit creates bfloat16 opt states
        report_to="none",
        dataloader_num_workers=0,
    )

    # Variants to try — ordered from newest TRL to oldest
    # max_length (new)        → TRL >= 0.12
    # max_seq_length (old)    → TRL 0.8–0.11
    # dataset_text_field      → needed only in some middle versions
    variants = [
        {"max_length": args.max_seq_len},
        {"max_seq_length": args.max_seq_len},
        {"max_length": args.max_seq_len, "dataset_text_field": "text"},
        {"max_seq_length": args.max_seq_len, "dataset_text_field": "text"},
        {},   # Absolute fallback — no extra params at all
    ]

    cfg = try_init(SFTConfig, base, variants)
    print(f"\n[5a] SFTConfig built successfully.")
    return cfg


# ==============================================================================
# STEP 5b — BUILD SFTTRAINER (version-agnostic)
# ==============================================================================

def build_trainer(model, dataset, sft_config, lora_config, tokenizer):
    """
    Build SFTTrainer in a way that works across TRL versions.

    Key API changes:
      - processing_class (new, >= TRL 0.12)  replaces  tokenizer (old)
      - peft_config passed here wraps the model — do NOT call get_peft_model()
        manually beforehand, or you will get:
        ValueError: You passed a PeftModel instance together with a peft_config
    """
    from trl import SFTTrainer

    base = dict(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=lora_config,   # SFTTrainer wraps model internally
    )

    # processing_class (new) → tokenizer (old) → no tokenizer kwarg
    variants = [
        {"processing_class": tokenizer},
        {"tokenizer": tokenizer},
        {},
    ]

    trainer = try_init(SFTTrainer, base, variants)
    print(f"[5b] SFTTrainer built successfully.")

    # Print trainable parameter count after PEFT wrapping
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in trainer.model.parameters())
    print(f"     Trainable params : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return trainer


# ==============================================================================
# MERGE LORA & EXPORT (run after training)
# ==============================================================================

def merge_and_save(args: argparse.Namespace, tokenizer: AutoTokenizer):
    """
    Merge LoRA adapter weights into the base model and save a full merged model.
    Uses PeftModel directly (not AutoPeftModelForCausalLM) to avoid the
    torchao version-check bug in PEFT >= 0.14 on Colab.
    """
    from peft import PeftModel

    banner("Merging LoRA adapter into base model ...")
    print(f"  Adapter dir  : {args.output_dir}")
    print(f"  Merged dir   : {args.merged_dir}")
    print("  Loading fresh base model on CPU (avoids VRAM overflow) ...")

    # Load the raw base model in float16 on CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Load and attach LoRA adapter
    print("  Attaching LoRA adapter ...")
    try:
        peft_model = PeftModel.from_pretrained(
            base_model,
            args.output_dir,
            torch_dtype=torch.float16,
        )
    except ImportError as e:
        if "torchao" in str(e):
            print("  Detected torchao version incompatibility. Falling back to AutoPeftModelForCausalLM...")
            from peft import AutoPeftModelForCausalLM
            peft_model = AutoPeftModelForCausalLM.from_pretrained(
                args.output_dir,
                device_map="cpu",
                torch_dtype=torch.float16,
            )
        else:
            raise e

    # Merge adapter weights into base model and unload adapter scaffolding
    print("  Merging weights ...")
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.merged_dir)
    tokenizer.save_pretrained(args.merged_dir)
    print(f"  Merged model saved to: {args.merged_dir}")

    print("""
Next step — Convert to GGUF (run in Colab):
  !git clone https://github.com/ggerganov/llama.cpp /content/llama.cpp
  !cd /content/llama.cpp && pip install -r requirements.txt
  !python /content/llama.cpp/convert_hf_to_gguf.py ./rtu_merged_model \\
      --outfile rtu_senior_tutor_fp16.gguf --outtype f16
  !./llama.cpp/llama-quantize rtu_senior_tutor_fp16.gguf \\
      rtu_senior_tutor_q4.gguf Q4_K_M

Then download rtu_senior_tutor_q4.gguf to your laptop and run:
  ollama create rtu-senior -f Modelfile
""")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()

    banner("RTU Senior Tutor — QLoRA Fine-Tuning")
    print(f"  Base model  : {args.base_model}")
    print(f"  Dataset     : {args.dataset_path}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Device      : {device_info()}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size} (x{args.grad_accum} grad_accum = {args.batch_size * args.grad_accum} effective)")
    print(f"  Max seq len : {args.max_seq_len}")

    if args.merge_only:
        # --- Merge-only mode: load tokenizer and merge existing adapter ---
        tokenizer = load_tokenizer(args.base_model)
        merge_and_save(args, tokenizer)
        return

    # --- Full training pipeline ---

    # 1. Tokenizer
    tokenizer = load_tokenizer(args.base_model)

    # 2. Base model (4-bit quantized)
    model = load_model(args.base_model)

    # 3. LoRA config (applied by SFTTrainer internally)
    lora_config = build_lora_config()

    # 4. Dataset
    dataset = load_and_format_dataset(args.dataset_path, tokenizer)

    # 5. SFTConfig
    banner("Step 5 — Configuring SFTTrainer & Starting Fine-Tuning")
    sft_config = build_sft_config(args)

    # 5b. SFTTrainer
    trainer = build_trainer(model, dataset, sft_config, lora_config, tokenizer)

    # 6. Train
    banner("Training started — watch for loss values every 5 steps")
    trainer.train()

    # 7. Save LoRA adapter
    banner("Saving LoRA adapter ...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"  Adapter saved to: {args.output_dir}")

    # 8. Auto-merge after training
    merge_and_save(args, tokenizer)

    banner("SUCCESS — RTU Senior Tutor fine-tuning complete!")
    print(f"""
Summary:
  LoRA adapter  -> {args.output_dir}/
  Merged model  -> {args.merged_dir}/

What to do next:
  1. Convert rtu_merged_model/ to GGUF (see instructions above)
  2. Download rtu_senior_tutor_q4.gguf to your laptop
  3. Run: ollama create rtu-senior -f Modelfile
  4. In backend .env: set LLM_PROVIDER=ollama and OLLAMA_MODEL=rtu-senior
  5. The Senior Tutor persona is now your local offline AI!
""")


if __name__ == "__main__":
    main()
