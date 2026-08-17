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

DIALOGUE_QUESTIONS = [
    {
        "topic": "1. Знакомство и самопрезентация",
        "prompt": "Привет! Расскажи о себе: кто ты, какие задачи умеешь решать и на какой архитектуре работаешь?"
    },
    {
        "topic": "2. Алгоритмы и Python",
        "prompt": "Напиши на Python функцию алгоритма Дейкстры (Dijkstra) для поиска кратчайшего пути во взвешенном графе с использованием heapq. Добавь комментарии."
    },
    {
        "topic": "3. Математика и теория множеств",
        "prompt": "Реши задачу: В группе 30 студентов. 18 изучают Python, 14 изучают C++, а 6 изучают оба языка. Сколько студентов не изучают ни Python, ни C++? Распиши формулу и шаги решения."
    },
    {
        "topic": "4. Логическое рассуждение",
        "prompt": "Загадка: У меня нет голоса, но я говорю; нет крыльев, но я летаю; нет зубов, но я могу укусить. Что я такое?"
    },
    {
        "topic": "5. Системная архитектура (Highload)",
        "prompt": "Как спроектировать масштабируемую систему очередей для обработки 100 000 задач в секунду? Опиши 3 ключевых компонента."
    }
]

async def run_dialogue_session():
    print("=" * 70)
    print(" 🚀 ДИАЛОГОВАЯ СЕССИЯ С ЛОКАЛЬНЫМ НЕЙРОСЕТЕВЫМ ИИ (AETHER ENGINE)")
    print("=" * 70)

    history = [
        ChatMessage(role="system", content="Ты — AETHER, высокоинтеллектуальный автономный AI-ассистент. Ты даешь точные, глубокие, структурированные ответы на русском языке с примерами кода и пошаговыми рассуждениями.")
    ]

    for idx, item in enumerate(DIALOGUE_QUESTIONS, 1):
        print(f"\n{'#' * 60}")
        print(f"📌 {item['topic']}")
        print(f"👤 Вопрос: {item['prompt']}")
        print(f"{'-' * 60}")
        print("🤖 Ответ AETHER: ", end="", flush=True)

        history.append(ChatMessage(role="user", content=item["prompt"]))
        
        t0 = time.time()
        tokens = []
        async for chunk in proprietary_engine.generate_stream(messages=history, model="aether-neural-local", max_tokens=300, temperature=0.6):
            if chunk.event_type == "token":
                print(chunk.delta_content, end="", flush=True)
                tokens.append(chunk.delta_content)

        elapsed = time.time() - t0
        full_answer = "".join(tokens)
        history.append(ChatMessage(role="assistant", content=full_answer))
        print(f"\n[⏱️ Время генерации: {elapsed:.2f} сек. | Токенов: ~{len(tokens)}]")

    print("\n" + "=" * 70)
    print(" 🎉 ДИАЛОГОВАЯ СЕССИЯ УСПЕШНО ЗАВЕРШЕНА!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_dialogue_session())
