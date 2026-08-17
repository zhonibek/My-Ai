import sys
import json
import httpx
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def test_live_chat_and_rag():
    print("==================================================================")
    print(" 🚀 TESTING LIVE RUNNING BACKEND SERVER (http://127.0.0.1:8000)")
    print("==================================================================")

    # 1. Health check
    r = httpx.get("http://127.0.0.1:8000/")
    print(f"1. Server Status: {r.status_code} -> {r.json()}")

    # 2. Test File Upload (Multipart Form)
    txt_content = "Проект Альтаир: Секретный ключ авторизации равен ALTAIR-998877."
    files = {"file": ("altair_doc.txt", txt_content.encode("utf-8"), "text/plain")}
    upload_res = httpx.post("http://127.0.0.1:8000/api/v1/files/upload", files=files, data={"user_id": "live_user"})
    print(f"2. File Upload Status: {upload_res.status_code} -> {upload_res.json()}")
    file_id = upload_res.json().get("file_id")

    # 3. Test Kazakh language & RAG in live stream
    print("\n3. Testing Streaming Query in Kazakh / Russian with attached file:")
    payload = {
        "messages": [
            {"role": "system", "content": "You are AETHER. Қазақша сауатты жауап бер."},
            {"role": "user", "content": "Сәлеметсіз бе! Жүктелген құжаттағы Альтаир жобасының құпия кілті қандай?"}
        ],
        "model": "aether-neural-local",
        "file_ids": [file_id] if file_id else [],
        "user_id": "live_user"
    }

    t0 = time.time()
    tokens = []
    with httpx.stream("POST", "http://127.0.0.1:8000/api/v1/chat/stream", json=payload, timeout=60.0) as resp:
        print(f"   Stream Connection: {resp.status_code}")
        for line in resp.iter_lines():
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    delta = data.get("delta", "")
                    if delta:
                        tokens.append(delta)
                        print(delta, end="", flush=True)
                except Exception:
                    pass

    elapsed = time.time() - t0
    print(f"\n\n[⏱️ Завершено за {elapsed:.2f} сек | Получено {len(tokens)} чанков]")
    print("==================================================================")

if __name__ == "__main__":
    test_live_chat_and_rag()
