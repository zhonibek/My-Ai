import json
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.providers.base import ChatMessage
from app.orchestrator.engine import orchestrator
from app.rag.vector_store import vector_store

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = Field("aether-neural-local", description="Target model ID or 'auto'")
    enable_web_search: bool = True
    file_ids: List[str] = Field(default_factory=list)
    user_id: str = "default_user"
    temperature: float = 0.7

@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    High-Performance FastAPI SSE Streaming Endpoint:
    Streams real-time tokens, reasoning steps, tool calls, and RAG document contexts.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty.")

    # Retrieve RAG context if user uploaded or attached files
    rag_context_text = ""
    if request.file_ids:
        # First attempt: direct file retrieval
        attached_chunks = await vector_store.get_chunks_by_file_ids(user_id=request.user_id, file_ids=request.file_ids)
        if not attached_chunks:
            # Fallback search
            last_prompt = request.messages[-1].content
            attached_chunks = await vector_store.search_similar(
                user_id=request.user_id,
                query=last_prompt,
                top_k=5,
                file_ids=request.file_ids
            )
        if attached_chunks:
            rag_context_text = "\n\n".join([f"[{c.file_name}]:\n{c.content}" for c in attached_chunks])

    async def event_generator():
        try:
            async for chunk in orchestrator.execute_chat_stream(
                messages=request.messages,
                model=request.model,
                file_ids=request.file_ids,
                enable_web_search=request.enable_web_search,
                user_id=request.user_id,
                rag_context=rag_context_text
            ):
                payload = {
                    "id": chunk.id,
                    "model": chunk.model,
                    "delta": chunk.delta_content,
                    "event_type": chunk.event_type,
                    "finish_reason": chunk.finish_reason,
                    "metadata": chunk.metadata
                }
                yield f"data: {json.dumps(payload)}\n\n"
                # Ultra-low latency yield (no artificial delay)
                await asyncio.sleep(0.001)
        except Exception as e:
            err_payload = {"event_type": "error", "error": str(e)}
            yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
