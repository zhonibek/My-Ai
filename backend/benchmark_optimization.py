import os
import sys
import asyncio
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure backend root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.providers.base import ChatMessage
from app.inference.local_engine import proprietary_engine, local_neural_engine

async def run_benchmark():
    print("==================================================================")
    print(" ⚡ [DEEP-OPTIMIZATION BENCHMARK] AETHER NEURAL ENGINE")
    print("==================================================================")

    print("\n[*] 1. WARM-UP & QUANTIZATION STATUS:")
    local_neural_engine.ensure_model_loaded()
    print(f" - Model Device: {local_neural_engine.device}")
    print(f" - Quantized Layers Active: {hasattr(local_neural_engine.model, 'forward')}")

    test_queries = [
        "Напиши функцию на Python для вычисления чисел Фибоначчи через генератор.",
        "Қазақстанның астанасы және оның басты көрнекті орындары туралы қысқаша айтып бер."
    ]

    total_tokens = 0
    total_time = 0.0

    for idx, q in enumerate(test_queries, 1):
        print(f"\n--- Benchmark Query {idx} ---")
        print(f"Prompt: {q}")
        messages = [
            ChatMessage(role="system", content="You are AETHER, a fast and articulate AI assistant. Answer concisely."),
            ChatMessage(role="user", content=q)
        ]

        t0 = time.time()
        tokens = []
        async for chunk in proprietary_engine.generate_stream(messages=messages, model="aether-neural-local", max_tokens=120, temperature=0.5):
            if chunk.event_type == "token":
                tokens.append(chunk.delta_content)

        elapsed = time.time() - t0
        gen_text = "".join(tokens)
        token_count = len(tokens)
        speed = token_count / max(elapsed, 0.01)

        total_tokens += token_count
        total_time += elapsed

        print(f"Response ({token_count} chunks, {elapsed:.2f}s, {speed:.2f} tokens/sec):")
        print(f"{gen_text[:200]}...")

    avg_speed = total_tokens / max(total_time, 0.01)
    print("\n==================================================================")
    print(f" 🚀 BENCHMARK RESULTS: Total Tokens: {total_tokens} | Total Time: {total_time:.2f}s | Avg Speed: {avg_speed:.2f} tok/s")
    print("==================================================================")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
