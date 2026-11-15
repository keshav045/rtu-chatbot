# RTU Senior Tutor: Free Google Colab Fine-Tuning Guide (Section 8)

This guide walks you through fine-tuning **Qwen2.5-1.5B-Instruct** (or Llama-3.2-1B) on the 200-example RTU exam seed dataset using **free Google Colab T4 GPU (16 GB VRAM)** in ~15 minutes.

---

## 1. Google Colab Setup

1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime > Change runtime type** and select **T4 GPU**.
3. Upload `seed_dataset_200.jsonl` and `train_qlora_colab.py` using the Files side panel.

---

## 2. Install Dependencies

In a new Colab cell, run:
```bash
!pip install -q torch transformers datasets trl peft bitsandbytes accelerate
```

---

## 3. Run QLoRA Training

Run the training script:
```bash
!python train_qlora_colab.py \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset_path "seed_dataset_200.jsonl" \
  --output_dir "./rtu_senior_tutor_lora" \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 2e-4
```

### Resource Profile on Colab T4:
- **VRAM Usage**: ~4.2 GB (out of 15.0 GB available)
- **Training Time**: ~12-14 minutes for 3 epochs (200 examples)
- **Trainable Parameters**: ~18M (out of 1.54B, ~1.1% LoRA footprint)

---

## 4. Merge LoRA & Export to GGUF for Ollama

After training finishes, merge the LoRA adapters into the base weights:

```python
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

output_dir = "./rtu_senior_tutor_lora"
merged_dir = "./rtu_senior_tutor_merged"

# Load and merge
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    device_map="auto",
    torch_dtype=torch.float16
)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_dir)

tokenizer = AutoTokenizer.from_pretrained(output_dir)
tokenizer.save_pretrained(merged_dir)
print("Model successfully merged into 16-bit weights!")
```

### Convert to GGUF using llama.cpp:
```bash
!git clone https://github.com/ggerganov/llama.cpp
!pip install -r llama.cpp/requirements.txt
!python llama.cpp/convert_hf_to_gguf.py ./rtu_senior_tutor_merged --outfile ./rtu_senior_tutor_q4.gguf --outtype q4_k_m
```

---

## 5. Deploy to Local Laptop via Ollama

Download `rtu_senior_tutor_q4.gguf` (~1.1 GB) to your laptop.

Create a `Modelfile`:
```dockerfile
FROM ./rtu_senior_tutor_q4.gguf
PARAMETER temperature 0.3
PARAMETER top_p 0.9
SYSTEM """You are Senior Tutor, a high-scoring RTU senior student. You explain concepts in English and Hinglish with 10-mark structures, diagrams, and citations."""
```

Register and run in Ollama:
```bash
ollama create rtu-senior -f Modelfile
ollama run rtu-senior
```

Now select `provider: "ollama"` in your RTU AI Academic Assistant, and it runs 100% locally on your CPU with zero external API calls!

