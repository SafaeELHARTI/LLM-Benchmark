import torch
import time
import csv
import os
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_DIR = "./results/lora_adapter"
PROMPT = "Explain what quantization means in 2 sentences."
MAX_NEW_TOKENS = 100
RESULTS_FILE = "./results/benchmark_results.csv"
writer = SummaryWriter(log_dir="./tensorboard_logs/benchmark")

def measure(model, tokenizer, label, step):
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    # warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)
    torch.cuda.synchronize()

    # measure
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    vram = torch.cuda.max_memory_allocated() / 1e9
    throughput = MAX_NEW_TOKENS / elapsed
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n--- {label} ---")
    print(f"Latency:    {elapsed:.2f}s")
    print(f"VRAM:       {vram:.2f} GB")
    print(f"Throughput: {throughput:.1f} tokens/s")
    print(f"Output:     {text[:100]}...")

    # Log to TensorBoard
    writer.add_scalar("Latency_s", elapsed, step)
    writer.add_scalar("VRAM_GB", vram, step)
    writer.add_scalar("Throughput_tokens_per_s", throughput, step)
    writer.add_text(label, text, step)

    return {
        "model": label,
        "latency_s": round(elapsed, 2),
        "vram_gb": round(vram, 2),
        "throughput_tokens_s": round(throughput, 1),
    }

def run_all():
    results = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    import gc

    # --- BF16 ---
    print("\n[1/4] Loading BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    results.append(measure(model, tokenizer, "BF16", step=0))
    del model; gc.collect(); torch.cuda.empty_cache()

    # --- INT8 ---
    print("\n[2/4] Loading INT8...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto"
    )
    results.append(measure(model, tokenizer, "INT8", step=1))
    del model; gc.collect(); torch.cuda.empty_cache()

    # --- INT4 ---
    print("\n[3/4] Loading INT4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    results.append(measure(model, tokenizer, "INT4", step=2))
    del model; gc.collect(); torch.cuda.empty_cache()

    # --- LoRA ---
    print("\n[4/4] Loading LoRA adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    results.append(measure(model, tokenizer, "LoRA (QLoRA)", step=3))
    del model; gc.collect(); torch.cuda.empty_cache()

    # --- Save CSV ---
    os.makedirs("./results", exist_ok=True)
    with open(RESULTS_FILE, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=results[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(results)

    print(f"\n✅ Results saved to {RESULTS_FILE}")

    # --- Summary table ---
    print("\n{'='*50}")
    print(f"{'Model':<15} {'Latency':>10} {'VRAM':>10} {'Throughput':>15}")
    print("-" * 50)
    for r in results:
        print(f"{r['model']:<15} {r['latency_s']:>9.2f}s {r['vram_gb']:>9.2f}GB {r['throughput_tokens_s']:>13.1f} t/s")

    writer.close()

if __name__ == "__main__":
    run_all()