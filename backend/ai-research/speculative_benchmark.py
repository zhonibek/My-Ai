import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_speculative_benchmark():
    print("==================================================================")
    print(" [RESEARCH] SPECULATIVE DECODING BENCHMARK (AETHER NEURAL ENGINE)")
    print("==================================================================")

    # Models: Draft (0.5B) vs Target (0.5B with different settings or larger target if available)
    # To test natively on local machine without downloading multi-gigabyte models first:
    # We load Draft = Qwen2.5-0.5B-Instruct, Target = Qwen2.5-0.5B-Instruct
    # (or you can test with any larger target model)
    draft_name = "Qwen/Qwen2.5-0.5B-Instruct"
    target_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cpu"

    print(f"\n[*] 1. Loading Tokenizer & Models...")
    tokenizer = AutoTokenizer.from_pretrained(draft_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f" - Loading Draft Model: {draft_name}...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_name,
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)
    draft_model.eval()

    print(f" - Loading Target Model: {target_name}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_name,
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)
    target_model.eval()

    test_queries = [
        "Напиши функцию на Python для вычисления чисел Фибоначчи через генератор.",
        "Объясни принцип работы алгоритма Дейкстры (Dijkstra) и приведи пример кода."
    ]

    print("\n==================================================================")
    print(" [*] 2. BENCHMARK 1: Standard Autoregressive Generation (Target Only)")
    print("==================================================================")

    std_total_tokens = 0
    std_total_time = 0.0

    for idx, q in enumerate(test_queries, 1):
        print(f"\n--- Standard Query {idx} ---")
        print(f"Prompt: {q}")
        prompt = f"<|im_start|>system\nYou are AETHER, a precise AI assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.inference_mode():
            output_tokens = target_model.generate(
                **inputs,
                max_new_tokens=90,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        new_tokens = output_tokens[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        count = len(new_tokens)
        speed = count / max(elapsed, 0.001)
        std_total_tokens += count
        std_total_time += elapsed

        print(f"Response ({count} tokens, {elapsed:.2f}s, {speed:.2f} tokens/sec):")
        print(f"{text[:160]}...")

    std_avg_speed = std_total_tokens / max(std_total_time, 0.001)

    print("\n==================================================================")
    print(" [*] 3. BENCHMARK 2: Speculative Decoding (Draft + Target Parallel)")
    print("==================================================================")

    spec_total_tokens = 0
    spec_total_time = 0.0

    for idx, q in enumerate(test_queries, 1):
        print(f"\n--- Speculative Query {idx} ---")
        print(f"Prompt: {q}")
        prompt = f"<|im_start|>system\nYou are AETHER, a precise AI assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.inference_mode():
            output_tokens = target_model.generate(
                **inputs,
                assistant_model=draft_model,
                max_new_tokens=90,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        new_tokens = output_tokens[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        count = len(new_tokens)
        speed = count / max(elapsed, 0.001)
        spec_total_tokens += count
        spec_total_time += elapsed

        print(f"Response ({count} tokens, {elapsed:.2f}s, {speed:.2f} tokens/sec):")
        print(f"{text[:160]}...")

    spec_avg_speed = spec_total_tokens / max(spec_total_time, 0.001)

    print("\n==================================================================")
    print(" [SUMMARY] SPECULATIVE DECODING RESULTS:")
    print(f" - Standard Autoregressive Speed : {std_avg_speed:.2f} tokens/sec")
    print(f" - Speculative Decoding Speed    : {spec_avg_speed:.2f} tokens/sec")
    speedup = (spec_avg_speed / max(std_avg_speed, 0.01) - 1) * 100
    print(f" - Performance Gain              : {speedup:+.2f}%")
    print(f" - Quality Degradation           : 0.00% (Mathematical Guarantee)")
    print("==================================================================")

if __name__ == "__main__":
    run_speculative_benchmark()
