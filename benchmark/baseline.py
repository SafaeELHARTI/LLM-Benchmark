import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Explain what quantization means in 2 sentences."

def run_baseline():
    print(f"Loading {MODEL_ID} in BF16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    # warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n--- Baseline (BF16) ---")
    print(f"Output: {text}")
    print(f"Latency: {elapsed:.2f}s")
    print(f"VRAM: {vram:.2f} GB")

if __name__ == "__main__":
    run_baseline()