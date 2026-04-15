import torch
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from peft import PeftModel

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./results/lora_adapter"

def run_lora():
    print("Loading model in INT4 + LoRA (QLoRA)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

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
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("tatsu-lab/alpaca", split="train[:500]")

    def format_prompt(example):
        return {"text": f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"}

    dataset = dataset.map(format_prompt)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        report_to="tensorboard",
        logging_dir="./tensorboard_logs"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=512,
    )

    print("Starting training...")
    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start
    vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nTraining time: {elapsed:.1f}s")
    print(f"Peak VRAM: {vram:.2f} GB")
    trainer.save_model(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")


def run_lora_inference():
    print('Loading base model + LoRA adapter...')
    
    # Free memory from training
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True  # allows CPU offload if needed
    )
    
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": 0,
        "lm_head": 0,
        "model.layers": 0,
    }
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    prompt = "### Instruction:\nExplain what LoRA is.\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    vram = torch.cuda.max_memory_allocated() / 1e9
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'Output: {text}')
    print(f'Latency: {elapsed:.2f}s | VRAM: {vram:.2f} GB')

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "inference":
        run_lora_inference()
    else:
        run_lora()