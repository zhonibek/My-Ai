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
from app.inference.local_engine import proprietary_engine

MULTILINGUAL_TESTS = [
    {
        "lang": "Қазақ тілі (Kazakh)",
        "prompt": "Сәлем! Өзің туралы қазақша айтып бер: сен кімсің және қандай тапсырмаларды орындай аласың?"
    },
    {
        "lang": "Қазақ тілі (Kazakh - Техникалық сұрақ)",
        "prompt": "Python тілінде тізімді (list) қалай сұрыптауға болады? Қысқаша мысал келтір."
    },
    {
        "lang": "Русский язык (Russian - Грамматика и рассуждение)",
        "prompt": "Привет! Объясни разницу между компилятором и интерпретатором. Приведи по 2 примера языков для каждого."
    }
]

async def run_multilingual_test():
    print("==================================================================")
    print(" 🌐 [MULTILINGUAL TEST] Қазақ тілі & Русский язык Evaluation")
    print("==================================================================")

    for item in MULTILINGUAL_TESTS:
        print(f"\n[{item['lang']}]")
        print(f"Сұрақ / Вопрос: {item['prompt']}")
        print("-" * 50)
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are AETHER, a polyglot AI assistant. "
                    "Қазақ тілінде сұрақ қойылса, міндетті түрде қазақ тілінде сауатты, таза әрі түсінікті жауап бер. "
                    "На русском языке отвечай безупречно грамотно и структурированно."
                )
            ),
            ChatMessage(role="user", content=item["prompt"])
        ]

        t0 = time.time()
        tokens = []
        async for chunk in proprietary_engine.generate_stream(messages=messages, model="aether-neural-local", max_tokens=150, temperature=0.6):
            if chunk.event_type == "token":
                tokens.append(chunk.delta_content)

        elapsed = time.time() - t0
        answer = "".join(tokens)
        print("Жауап / Ответ:")
        print(answer)
        print(f"[⏱️ {elapsed:.2f} сек | {len(tokens)} токенов | {len(tokens)/max(elapsed, 0.01):.1f} tok/s]")
        assert len(answer) > 15, "Answer must not be empty"

    print("\n[SUCCESS] Multilingual evaluation passed successfully!")

if __name__ == "__main__":
    asyncio.run(run_multilingual_test())
