import json
import uuid
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional
from app.providers.base import ModelProvider, ChatMessage, CompletionResponse, StreamChunk, ModelInfo
from app.config import settings

class GoogleProvider(ModelProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GOOGLE_API_KEY

    @property
    def provider_name(self) -> str:
        return "google"

    async def list_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                id="gemini-1-5-pro",
                name="Gemini 1.5 Pro",
                provider="google",
                context_window=1000000,
                capabilities=["chat", "long_context", "multimodal", "coding", "tools"],
                description="Google's flagship 1M+ token context multimodal model"
            ),
            ModelInfo(
                id="gemini-2-0-flash",
                name="Gemini 2.0 Flash",
                provider="google",
                context_window=1000000,
                capabilities=["chat", "fast", "multimodal", "tools"],
                description="Ultra-fast high performance multimodal model"
            )
        ]

    async def generate(
        self,
        messages: List[ChatMessage],
        model: str = "gemini-1-5-pro",
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None
    ) -> CompletionResponse:
        content = f"[Google Gemini Response ({model})]: Answer processed with massive context window."
        return CompletionResponse(
            id=f"gemini-{uuid.uuid4().hex[:8]}",
            content=content,
            model=model,
            provider="google",
            prompt_tokens=100,
            completion_tokens=40,
            finish_reason="STOP"
        )

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        model: str = "gemini-1-5-pro",
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        req_id = f"gemini-{uuid.uuid4().hex[:8]}"
        mock_text = f"Response from **Google Gemini** (`{model}`).\n\n- Context Capacity: 1,000,000+ tokens\n- Multimodal Processing: Enabled\n- Web Search Integration: Ready"
        words = mock_text.split(" ")
        for i, word in enumerate(words):
            yield StreamChunk(id=req_id, model=model, delta_content=word + (" " if i < len(words)-1 else ""), event_type="token")
        yield StreamChunk(id=req_id, model=model, delta_content="", finish_reason="STOP", event_type="done")
