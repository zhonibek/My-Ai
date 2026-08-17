import os
import sys
import asyncio
from httpx import AsyncClient, ASGITransport

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure backend root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app
from app.rag.vector_store import vector_store

async def test_file_and_image_rag():
    print("==================================================================")
    print(" 📁 [RAG & FILE UPLOAD TEST] Text Documents & Visual Assets")
    print("==================================================================")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Test Text File Upload (Multipart Form-Data)
        sample_txt = "Project Apollo Alpha 2026: The secret system codename is Quantum-Eagle. Primary target revenue is $50M."
        files_text = {
            "file": ("apollo_spec.txt", sample_txt.encode("utf-8"), "text/plain")
        }
        res_txt = await ac.post("/api/v1/files/upload", files=files_text, data={"user_id": "test_user"})
        print("1. Text File Upload Status:", res_txt.status_code)
        data_txt = res_txt.json()
        print("   Upload Response:", data_txt)
        assert res_txt.status_code == 200
        assert data_txt["chunks_indexed"] > 0
        txt_file_id = data_txt["file_id"]

        # 2. Test Image Upload (Multipart Form-Data)
        sample_img_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x02\x00\x00\x00\x90\x91h6"
        files_img = {
            "file": ("architecture_diagram.png", sample_img_bytes, "image/png")
        }
        res_img = await ac.post("/api/v1/files/upload", files=files_img, data={"user_id": "test_user"})
        print("\n2. Image Upload Status:", res_img.status_code)
        data_img = res_img.json()
        print("   Image Upload Response:", data_img)
        assert res_img.status_code == 200
        img_file_id = data_img["file_id"]

        # 3. Test Vector Store Retrieval
        retrieved_chunks = await vector_store.get_chunks_by_file_ids("test_user", [txt_file_id, img_file_id])
        print(f"\n3. Retrieved {len(retrieved_chunks)} chunks directly from Vector Store:")
        for c in retrieved_chunks:
            print(f" - [{c.file_name}] (ID: {c.file_id}): {c.content[:80]}...")

        assert len(retrieved_chunks) == 2, "Expected 2 indexed chunks (1 text doc, 1 image)"

        # 4. Test RAG-Augmented Chat Stream
        print("\n4. Testing AI Answer generation with attached file context...")
        chat_payload = {
            "messages": [{"role": "user", "content": "What is the secret system codename in the uploaded Apollo file?"}],
            "model": "aether-neural-local",
            "file_ids": [txt_file_id],
            "user_id": "test_user"
        }
        chat_res = await ac.post("/api/v1/chat/stream", json=chat_payload)
        print("   Chat Stream Status:", chat_res.status_code)
        assert chat_res.status_code == 200

        print("\n[SUCCESS] File & Image RAG Upload Pipeline tested and working 100% cleanly!")

if __name__ == "__main__":
    asyncio.run(test_file_and_image_rag())
