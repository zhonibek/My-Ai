import sys
import os
import asyncio

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.inference.local_engine import proprietary_engine
from app.providers.base import ChatMessage

TEST_CASES = [
    # 1. Greetings
    ("Greeting - RU", "привет"),
    ("Greeting - EN", "hi"),
    ("Short Query", "what"),
    
    # 2. AI Research Engineer
    ("AI Research", "Создай нейросеть для обработки текста"),
    
    # 3. Math & Logic
    ("Math - Simple", "2+2"),
    ("Logic - Riddle", "У отца Ивана 5 сыновей: 1. Чач, 2. Чеч, 3. Чич, 4. Чоч. Как зовут 5-го сына?"),

    # 4. Identity
    ("Identity", "Кто ты и какова твоя миссия?"),
]

async def run_benchmark():
    print("================================================================")
    print("  AETHER ENGINE v5.0 (AI RESEARCH ENGINEER EVALUATION)")
    print("================================================================\n")

    for category, prompt in TEST_CASES:
        print(f"------------ [{category}] ------------")
        print(f"Prompt: {prompt}")
        messages = [ChatMessage(role="user", content=prompt)]
        
        try:
            response = await proprietary_engine.generate(messages, model="aether-7b-custom")
            output = response.content.strip()
            print(f"Output:\n{output}\n")
        except Exception as e:
            print(f"ERROR: {e}\n")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
