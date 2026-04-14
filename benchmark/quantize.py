import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Explain what quantization means in 2 sentences."

def run_int8():
    print(f"Loading {MODEL_ID} in INT8...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)  # warmup

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n--- INT8 Quantization ---")
    print(f"Output: {text}")
    print(f"Latency: {elapsed:.2f}s")
    print(f"VRAM: {vram:.2f} GB")

def run_int4():
    print(f"Loading {MODEL_ID} in INT4...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)  # warmup

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n--- INT4 Quantization ---")
    print(f"Output: {text}")
    print(f"Latency: {elapsed:.2f}s")
    print(f"VRAM: {vram:.2f} GB")

if __name__ == "__main__":
    run_int8()
    run_int4()