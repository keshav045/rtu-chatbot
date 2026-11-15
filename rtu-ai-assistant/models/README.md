# Local LLM Models

This directory contains configuration files and local GGUF quantizations for the RTU Senior Tutor assistant.

## GGUF Model Placement
Place your quantized GGUF file in this directory:
- **Filename:** `rtu_senior_tutor_q4.gguf`
- **Recommended Base:** Qwen2.5-1.5B-Instruct-Q4_K_M or Llama-3.2-3B-Instruct-Q4_K_M fine-tuned / prompted for RTU engineering.

## Ollama Setup
To create and run the custom model via Ollama:
```bash
ollama create rtu-senior-tutor -f Modelfile
ollama run rtu-senior-tutor
```

> **Note:** Binary weights (`*.gguf`) exceed 100 MB and are excluded from Git tracking via `.gitignore`.
