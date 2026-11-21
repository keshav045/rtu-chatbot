# RTU Senior Tutor: Fine-Tuning Guide (Expanded Dataset)

This guide walks you through fine-tuning **Qwen2.5-1.5B-Instruct** into the **RTU Senior Tutor** persona using the expanded **748-example dataset** (`rtu_train_dataset_expanded.jsonl`) generated directly from 1,534 authentic RTU past exam questions across all 8 semesters (22 subjects).

---

## 1. Dataset Overview

- **Source File**: `rtu_train_dataset_expanded.jsonl`
- **Total Training Examples**: 748
- **Core Tasks Covered**:
  - `explain_topic` (436 examples / 58.3%): 10-mark model answers (Definition, Examiner Diagram Guide, Working Steps, Example/Complexity, RTU Scoring Tips) & 2-mark brevity in English & Hinglish.
  - `quiz_generation` (106 examples / 14.2%): 5-question multi-choice and short-answer drills calibrated to RTU marking scheme.
  - `structured_extraction` (96 examples / 12.8%): Raw OCR question paper text extraction to structured JSON.
  - `answer_grading` (60 examples / 8.0%): 0-10 scoring rubric breakdown with senior feedback.
  - `refusal_idk` (50 examples / 6.7%): Anti-hallucination guardrails refusing non-academic / out-of-syllabus requests.

To regenerate or customize the dataset from the live SQLite database:
```bash
python generate_expanded_dataset.py
```

---

## 2. Option A: Free Google Colab T4 GPU (Recommended - ~15 Mins)

### Step 1: Open Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **Runtime > Change runtime type** and select **T4 GPU**.
3. In the left Files panel, upload:
   - `train_qlora_colab.py`
   - `rtu_train_dataset_expanded.jsonl`

### Step 2: Install Dependencies
In a new Colab code cell:
```bash
!pip install -q torch transformers datasets trl peft bitsandbytes accelerate
```

### Step 3: Run Training
```bash
!python train_qlora_colab.py \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset_path "rtu_train_dataset_expanded.jsonl" \
  --output_dir "./rtu_lora_adapter" \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 2e-4 \
  --max_seq_len 2048
```
- **VRAM Usage**: ~5.2 GB (well within Colab T4's 15 GB).
- **Training Time**: ~15–18 minutes.

### Step 4: Merge LoRA & Export
```bash
!python train_qlora_colab.py --merge_only --output_dir "./rtu_lora_adapter" --merged_dir "./rtu_merged_model"
```

### Step 5: Convert to GGUF for Ollama
```bash
!git clone https://github.com/ggerganov/llama.cpp
!pip install -r llama.cpp/requirements.txt
!python llama.cpp/convert_hf_to_gguf.py ./rtu_merged_model --outfile ./rtu_senior_tutor_q4.gguf --outtype q4_k_m
```
Download `rtu_senior_tutor_q4.gguf` to your laptop.

---

## 3. Option B: Local PC Fine-Tuning (4 GB VRAM - GTX 1650)

If you prefer to train locally on your machine:
```bash
pip install torch transformers datasets trl peft bitsandbytes accelerate
python train_qlora_local.py --dataset_path rtu_train_dataset_expanded.jsonl --epochs 3
```
- **VRAM Profile**: Uses ~3.4 GB VRAM via 4-bit NF4 quantization, gradient checkpointing, batch size 1, and gradient accumulation 16.

---

## 4. Deploying into Local Ollama

Once you have `rtu_senior_tutor_q4.gguf` on your machine:

1. Create a file named `Modelfile` in the same directory:
```dockerfile
FROM ./rtu_senior_tutor_q4.gguf
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
SYSTEM """You are Senior Tutor - an empathetic, high-scoring senior student from Rajasthan Technical University (RTU). You explain concepts in simple English or Hinglish, provide exam-oriented 10-mark structures (Definition, Diagram, Steps, Example, RTU Tip), grade student answers honestly with constructive feedback, and strictly refuse out-of-syllabus topics."""
```

2. Register and start model in Ollama:
```bash
ollama create rtu-senior -f Modelfile
ollama run rtu-senior
```

3. In the RTU AI Academic Assistant UI or `config.py`:
   - Set `OLLAMA_MODEL = "rtu-senior"`
   - The assistant will now answer using your custom fine-tuned model running 100% locally on your machine!
