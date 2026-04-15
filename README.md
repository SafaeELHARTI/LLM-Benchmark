# 🚀 LLM Optimization Benchmark — TinyLlama-1.1B

> Quantization & LoRA fine-tuning benchmark on a 4GB consumer GPU  
> **Tools:** Docker · PyTorch · bitsandbytes · PEFT · TensorBoard  
> **Model:** [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)  
> **Hardware:** NVIDIA GeForce RTX 3050 (4GB VRAM)

---

## 📊 Benchmark Results

| Model Variant | Latency | VRAM | Throughput | vs BF16 |
|---|---|---|---|---|
| BF16 (baseline) | 4.87s | 2.23 GB | 20.5 t/s | — |
| INT8 | 21.31s | 1.33 GB | 4.7 t/s | 4.4x slower, 40% less VRAM |
| INT4 (NF4) | 0.12s | 0.84 GB | 860.2 t/s | **40x faster**, 62% less VRAM |
| LoRA / QLoRA | 9.89s | 0.84 GB | 10.1 t/s | Fine-tuned adapter, same VRAM as INT4 |

---

## 🔍 Key Findings

- **INT4 is the clear winner** on this hardware — 40x faster than BF16 with 62% less VRAM, thanks to NF4 kernel optimizations in bitsandbytes
- **INT8 is surprisingly slow** — dequantization overhead at inference time outweighs the memory savings on small GPUs (4GB)
- **QLoRA makes fine-tuning possible on a 4GB GPU** — only 0.10% of parameters are trained (1.1M out of 1.1B), peak VRAM stays at 4.75 GB during training
- **LoRA inference adds overhead** vs raw INT4 — adapter loading and merging costs ~9.8s vs 0.12s

---

## 🏗️ Project Structure

```
llm-benchmark/
├── benchmark/
│   ├── baseline.py          # BF16 reference measurement
│   ├── quantize.py          # INT8 and INT4 quantization
│   ├── lora_finetune.py     # QLoRA fine-tuning + inference
│   └── run_benchmark.py     # Full benchmark harness (all variants)
├── results/
│   ├── benchmark_results.csv
│   └── lora_adapter/        # Saved LoRA adapter weights
├── tensorboard_logs/        # TensorBoard event files
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## ⚙️ How to Reproduce

### Prerequisites
- Windows 10/11 with WSL2 + Ubuntu
- NVIDIA GPU (≥4GB VRAM) with updated drivers
- Docker Desktop with NVIDIA Container Toolkit
- HuggingFace account + access token

### Setup

```bash
git clone https://github.com/SafaeElHarti/llm-benchmark.git
cd llm-benchmark
```

Create a `.env` file:
```
HF_TOKEN=your_huggingface_token_here
```

Build the Docker image:
```bash
docker compose build
```

### Run Benchmarks

**Baseline (BF16):**
```bash
docker compose run --rm benchmark python benchmark/baseline.py
```

**Quantization (INT8 + INT4):**
```bash
docker compose run --rm benchmark python benchmark/quantize.py
```

**QLoRA Fine-tuning:**
```bash
docker compose run --rm benchmark python benchmark/lora_finetune.py
```

**LoRA Inference:**
```bash
docker compose run --rm benchmark python benchmark/lora_finetune.py inference
```

**Full Benchmark (all variants):**
```bash
docker compose run --rm benchmark python benchmark/run_benchmark.py
```

### TensorBoard Dashboard

```bash
docker compose up tensorboard
```
Open [http://localhost:6007](http://localhost:6007)

---

## 📈 TensorBoard Charts

The benchmark logs latency, VRAM, and throughput for all variants to TensorBoard:

- `Latency_s` — inference time per variant (steps 0–3 = BF16, INT8, INT4, LoRA)
- `VRAM_GB` — peak GPU memory usage
- `Throughput_tokens_per_s` — tokens generated per second

---

## 🧠 What is QLoRA?

QLoRA combines INT4 quantization with LoRA fine-tuning:

1. The base model is loaded in **4-bit** (saving VRAM)
2. Small trainable matrices (**rank r=8**) are injected into attention layers
3. Only those matrices are trained — **0.10% of total parameters**
4. The original weights stay frozen

This makes fine-tuning a 1.1B model possible on a 4GB consumer GPU in under 2 minutes.

---

## 📦 Dependencies

```
transformers==4.40.0
accelerate==0.29.0
bitsandbytes==0.43.1
peft==0.10.0
trl==0.8.6
datasets==2.19.0
tensorboard==2.16.2
rich
```

---

## 👩‍💻 Author

**Safae ElHarti** — LLM Optimization Benchmark TP