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

async def test_neural_engine():
    print("==================================================================")
    print(" [*] TESTING LOCAL PROPRIETARY NEURAL ENGINE (AETHER)")
    print("==================================================================")

    print("\n--- 1. Testing Model Info & Registration ---")
    models = await proprietary_engine.list_models()
    for m in models:
        print(f" - [{m.provider}] {m.id} : {m.name}")
    assert any(m.id == "aether-neural-local" for m in models), "aether-neural-local must be registered"

    print("\n--- 2. Testing Synchronous Neural Generation (Russian Query) ---")
    messages = [
        ChatMessage(role="system", content="Ты — умный AI-ассистент AETHER. Отвечай связно, вежливо и по делу."),
        ChatMessage(role="user", content="Привет! Объясни простыми словами, что такое рекурсия в программировании, и приведи короткий пример.")
    ]
    t0 = time.time()
    response = await proprietary_engine.generate(messages=messages, model="aether-neural-local", max_tokens=150)
    elapsed = time.time() - t0

    print(f"Time taken: {elapsed:.2f}s")
    print(f"Finish reason: {response.finish_reason}")
    print(f"Response:\n{response.content}\n")
    assert len(response.content) > 20, "Response should be non-trivial"

    print("\n--- 3. Testing Streaming Generation (English Query) ---")
    messages_stream = [
        ChatMessage(role="system", content="You are AETHER, a helpful and articulate AI assistant."),
        ChatMessage(role="user", content="What is the difference between a stack and a queue? Give 1 sentence for each.")
    ]
    t0 = time.time()
    tokens = []
    print("Streaming tokens: ", end="", flush=True)
    async for chunk in proprietary_engine.generate_stream(messages=messages_stream, model="aether-neural-local", max_tokens=100):
        if chunk.event_type == "token":
            print(chunk.delta_content, end="", flush=True)
            tokens.append(chunk.delta_content)

    full_stream_text = "".join(tokens)
    print(f"\nStream completed in {time.time() - t0:.2f}s ({len(tokens)} chunks)")
    assert len(full_stream_text) > 10, "Stream output should not be empty"

    print("\n[SUCCESS] All Local Proprietary Neural Engine tests passed cleanly!")

if __name__ == "__main__":
    asyncio.run(test_neural_engine())
