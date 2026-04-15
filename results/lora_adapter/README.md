---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
tags:
  - lora
  - qlora
  - tinyllama
  - instruction-tuning
  - causal-lm
  - 4bit
license: apache-2.0
---

# TinyLlama-1.1B — QLoRA Adapter (Alpaca-500)

Fine-tuned LoRA adapter for [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) using QLoRA on 500 instruction-following examples from the Alpaca dataset.

This adapter was produced as part of an LLM optimization benchmark studying the latency/quality tradeoff of quantization and LoRA fine-tuning on a 4GB consumer GPU.

---

## Model Details

### Model Description

- **Developed by:** Safae ElHarti
- **Model type:** Causal Language Model (LoRA adapter)
- **Language:** English
- **License:** Apache 2.0
- **Base model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning method:** QLoRA (INT4 base + BF16 LoRA adapters)
- **Repository:** [SafaeElHarti/llm-benchmark](https://github.com/SafaeElHarti/llm-benchmark)

---

## Uses

### Direct Use

This adapter is intended for instruction-following tasks in English. Load it on top of the INT4-quantized TinyLlama base model for memory-efficient inference.

### Out-of-Scope Use

- Not suitable for production or safety-critical applications
- Not evaluated on multilingual tasks
- May hallucinate or produce incorrect answers — this is a lightweight research adapter trained on 500 examples

---

## How to Get Started

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_ID = "results/lora_adapter"  # or your HF Hub path

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

prompt = "### Instruction:\nExplain what LoRA is.\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Training Details

### Training Data

- **Dataset:** [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
- **Subset:** First 500 examples (`train[:500]`)
- **Format:** Instruction/Response pairs

```
### Instruction:
<instruction text>
### Response:
<response text>
```

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Training regime | BF16 mixed precision |
| Epochs | 1 |
| Per device batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Learning rate | 2e-4 |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, v_proj |
| Max sequence length | 512 |
| Base model precision | INT4 (NF4) |

### Training Results

| Metric | Value |
|---|---|
| Training time | ~107s (~1.8 min) |
| Peak VRAM | 4.75 GB |
| Final training loss | 1.687 |
| Trainable parameters | 1,126,400 (0.10% of total) |
| Hardware | NVIDIA GeForce RTX 3050 (4GB) |

**Loss curve:**

| Step | Epoch | Loss |
|---|---|---|
| 10 | 0.16 | 2.083 |
| 20 | 0.32 | 1.736 |
| 30 | 0.48 | 1.649 |
| 40 | 0.64 | 1.646 |
| 50 | 0.80 | 1.471 |
| 60 | 0.96 | 1.594 |

---

## Evaluation

### Inference Results

| Metric | Value |
|---|---|
| Inference latency | 17.12s |
| Inference VRAM | 0.84 GB |
| Throughput | 10.1 tokens/s |

### Benchmark Comparison

| Variant | Latency | VRAM | Throughput |
|---|---|---|---|
| BF16 baseline | 4.87s | 2.23 GB | 20.5 t/s |
| INT8 | 21.31s | 1.33 GB | 4.7 t/s |
| INT4 | 0.12s | 0.84 GB | 860.2 t/s |
| **LoRA (this adapter)** | **17.12s** | **0.84 GB** | **10.1 t/s** |

---

## Technical Specifications

### Model Architecture

- **Base:** TinyLlama-1.1B (22 transformer layers, 32 attention heads)
- **LoRA injection:** q_proj and v_proj in all attention layers
- **Adapter size:** ~4.5 MB (vs 2.2 GB for full BF16 model)

### Compute Infrastructure

- **Hardware:** NVIDIA GeForce RTX 3050 4GB (WSL2)
- **Software:** PyTorch 2.4.0 + CUDA 12.4, bitsandbytes 0.43.1, PEFT 0.10.0, TRL 0.8.6

---

## Framework Versions

- PEFT 0.10.0
- Transformers 4.40.0
- PyTorch 2.4.0
- bitsandbytes 0.43.1
- TRL 0.8.6