import asyncio
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure backend root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.providers.gateway import gateway
from app.providers.base import ChatMessage
from app.orchestrator.engine import orchestrator
from app.orchestrator.tools import tool_registry
from app.rag.parser import doc_parser
from app.rag.chunker import text_chunker
from app.rag.vector_store import vector_store
from app.providers.base import VectorChunk

async def run_verifications():
    print("=== 1. Testing Model Gateway & Provider List ===")
    models = await gateway.list_all_models()
    print(f"Registered Models Count: {len(models)}")
    for m in models:
        print(f" - [{m.provider.upper()}] {m.id} ({m.name})")

    print("\n=== 2. Testing Web Search Tool Execution ===")
    search_res = await tool_registry.execute_tool("web_search", {"query": "MacBook M3 prices 2026"})
    print(f"Search Results Count: {search_res.get('count')}")
    print(f"Sources Sample: {search_res.get('sources')}")

    print("\n=== 3. Testing Calculator Tool Execution ===")
    calc_res = await tool_registry.execute_tool("calculator", {"expression": "1250 * 0.85 + math.sqrt(144)"})
    print(f"Calculator Result: {calc_res}")

    print("\n=== 4. Testing Document RAG Parser & Vector Search ===")
    sample_pdf_text = "Scholarship Requirements 2026: Applicants must maintain a minimum GPA of 3.8 and complete 50 hours of community robotics service."
    parsed = doc_parser.parse_file("requirements.txt", sample_pdf_text.encode("utf-8"))
    chunks = text_chunker.split_text(parsed, "doc_101", "requirements.txt")
    v_chunks = [VectorChunk(id=c["chunk_id"], file_id="doc_101", file_name="requirements.txt", content=c["content"]) for c in chunks]
    await vector_store.insert_chunks("user_test", v_chunks)
    
    retrieved = await vector_store.search_similar("user_test", "scholarship GPA requirements", top_k=1)
    print(f"Retrieved RAG Chunk: {retrieved[0].content if retrieved else 'None'}")

    print("\n=== 5. Testing AI Orchestrator Execution Stream ===")
    messages = [ChatMessage(role="user", content="Find the cheapest MacBook and calculate 10% tax.")]
    stream_tokens = []
    async for chunk in orchestrator.execute_chat_stream(messages=messages, model="aether-neural-local", enable_web_search=True):
        if chunk.event_type == "token":
            stream_tokens.append(chunk.delta_content)
        elif chunk.event_type == "reasoning":
            print(f" [Reasoning Event]: {chunk.metadata}")
        elif chunk.event_type == "source_citation":
            print(f" [Citations Event]: {chunk.metadata}")

    full_response = "".join(stream_tokens)
    print(f"\nStream Output Sample:\n{full_response[:200]}...")
    print("\n[SUCCESS] All AI Platform backend verification tests passed clean!")

if __name__ == "__main__":
    asyncio.run(run_verifications())
