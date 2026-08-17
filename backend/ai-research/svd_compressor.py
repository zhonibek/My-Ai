import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from typing import Dict, Any, List

# Ensure backend root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from app.providers.base import ChatMessage

class SVDLinearApproximation(nn.Module):
    """
    Approximates a target Linear layer W (M x N) with two low-rank Linear layers:
    W_1 (k x N) and W_2 (M x k) such that W ≈ W_2 * W_1
    """
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))


def compress_model_svd(model: nn.Module, compression_ratio: float = 0.5) -> nn.Module:
    """
    Iterate over the model, find projection linear layers, and apply low-rank SVD factorization.
    """
    print(f"\n[*] Starting SVD compression with ratio: {compression_ratio:.2f}...")
    
    total_original_params = sum(p.numel() for p in model.parameters())
    layers_compressed = 0
    
    # Target projection layers in transformers
    target_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if layer name matches target projections
            if any(target in name for target in target_names):
                # Calculate target rank k
                m, n = module.weight.shape  # out_features, in_features
                max_rank = min(m, n)
                rank = int(max_rank * compression_ratio)
                rank = max(1, rank)
                
                # SVD decomposition: W = U * S * V^T
                with torch.no_grad():
                    # weight is of shape (out_features, in_features)
                    W = module.weight.data.float()
                    U, S, V = torch.linalg.svd(W, full_matrices=False)
                    
                    # Truncate to rank k
                    U_k = U[:, :rank]
                    S_k = S[:rank]
                    V_k = V[:rank, :]
                    
                    # W_1 = diag(S_k) * V_k
                    W1 = torch.diag(S_k) @ V_k
                    # W_2 = U_k
                    W2 = U_k
                    
                    # Create SVD approximation module
                    has_bias = module.bias is not None
                    svd_layer = SVDLinearApproximation(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=rank,
                        bias=has_bias
                    )
                    
                    # Copy weights
                    svd_layer.linear1.weight.data.copy_(W1.to(module.weight.dtype))
                    svd_layer.linear2.weight.data.copy_(W2.to(module.weight.dtype))
                    if has_bias:
                        svd_layer.linear2.bias.data.copy_(module.bias.data)
                
                # Replace module in model
                # Navigate path to replace the attribute
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], svd_layer)
                layers_compressed += 1
                
    total_compressed_params = sum(p.numel() for p in model.parameters())
    reduction = (1 - total_compressed_params / total_original_params) * 100
    print(f"[SUCCESS] SVD Compression Completed!")
    print(f" - Layers modified: {layers_compressed}")
    print(f" - Original params: {total_original_params:,}")
    print(f" - Compressed params: {total_compressed_params:,}")
    print(f" - Parameter Reduction: {reduction:.2f}%")
    return model


def run_benchmark(model, tokenizer, device="cpu"):
    test_queries = [
        "Напиши функцию на Python для вычисления чисел Фибоначчи через генератор.",
        "Қазақстанның астанасы және оның басты көрнекті орындары туралы қысқаша айтып бер."
    ]
    
    total_tokens = 0
    total_time = 0.0
    
    for idx, q in enumerate(test_queries, 1):
        print(f"\n--- Benchmark Query {idx} ---")
        print(f"Prompt: {q}")
        
        prompt = f"<|im_start|>system\nYou are AETHER, a fast and articulate AI assistant. Answer concisely.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        
        t0 = time.time()
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        
        new_tokens = output_tokens[0][input_len:]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        token_count = len(new_tokens)
        speed = token_count / max(elapsed, 0.01)
        
        total_tokens += token_count
        total_time += elapsed
        
        print(f"Response ({token_count} tokens, {elapsed:.2f}s, {speed:.2f} tokens/sec):")
        print(f"{gen_text[:200]}...")
        
    avg_speed = total_tokens / max(total_time, 0.01)
    print("\n==================================================================")
    print(f" [BENCHMARK RESULTS] Avg Speed: {avg_speed:.2f} tok/s | Total Time: {total_time:.2f}s")
    print("==================================================================")
    return avg_speed, total_time


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cpu"
    
    print("==================================================================")
    print(" [RESEARCH] SVD LOW-RANK APPROXIMATION EXPERIMENT")
    print("==================================================================")
    
    # 1. Load baseline model
    print(f"\n[*] Loading Baseline Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    
    print("\n[*] Running BASELINE BENCHMARK...")
    base_speed, base_time = run_benchmark(model, tokenizer, device)
    
    # 2. Apply SVD compression (compression ratio: 0.6 = keep 60% of singular values rank)
    compressed_model = compress_model_svd(model, compression_ratio=0.6)
    compressed_model.eval()
    
    print("\n[*] Running COMPRESSED MODEL BENCHMARK...")
    comp_speed, comp_time = run_benchmark(compressed_model, tokenizer, device)
    
    print("\n==================================================================")
    print(" [SUMMARY] COMPARISON SUMMARY:")
    print(f" - Baseline Speed: {base_speed:.2f} tokens/sec")
    print(f" - SVD-Compressed Speed: {comp_speed:.2f} tokens/sec")
    print(f" - Speedup: {(comp_speed / max(base_speed, 0.01) - 1)*100:+.2f}%")
    print("==================================================================")
