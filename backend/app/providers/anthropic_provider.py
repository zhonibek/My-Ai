import json
import uuid
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional
from app.providers.base import ModelProvider, ChatMessage, CompletionResponse, StreamChunk, ModelInfo
from app.config import settings

class AnthropicProvider(ModelProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.base_url = "https://api.anthropic.com/v1"

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def list_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                id="claude-3-5-sonnet",
                name="Claude 3.5 Sonnet",
                provider="anthropic",
                context_window=200000,
                capabilities=["chat", "coding", "reasoning", "vision", "tools"],
                description="Industry leading model for complex reasoning, coding, and writing"
            ),
            ModelInfo(
                id="claude-3-opus",
                name="Claude 3 Opus",
                provider="anthropic",
                context_window=200000,
                capabilities=["chat", "reasoning", "coding"],
                description="Anthropic's most powerful intelligence for deep analysis"
            )
        ]

    async def generate(
        self,
        messages: List[ChatMessage],
        model: str = "claude-3-5-sonnet",
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = 4096
    ) -> CompletionResponse:
        if not self.api_key:
            content = f"[Mock Anthropic Response ({model})]: Analytical response synthesized across {len(messages)} turns."
            return CompletionResponse(
                id=f"msg_{uuid.uuid4().hex[:8]}",
                content=content,
                model=model,
                provider="anthropic",
                prompt_tokens=len(str(messages)) // 4,
                completion_tokens=len(content) // 4,
                finish_reason="end_turn"
            )
            
        content = f"Response from Anthropic API ({model})"
        return CompletionResponse(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            content=content,
            model=model,
            provider="anthropic",
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="end_turn"
        )

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        model: str = "claude-3-5-sonnet",
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = 4096
    ) -> AsyncGenerator[StreamChunk, None]:
        req_id = f"msg_{uuid.uuid4().hex[:8]}"
        
        if not self.api_key:
            mock_text = "Greetings! This is a stream from **Anthropic Claude** (`" + str(model) + "`).\n\n1. **Deep Reasoning**: Complete\n2. **Code Synthesizer**: Ready\n3. **Context Length**: Up to 200k tokens."
            words = mock_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words)-1 else "")
                yield StreamChunk(
                    id=req_id,
                    model=model,
                    delta_content=chunk,
                    event_type="token"
                )
            yield StreamChunk(id=req_id, model=model, delta_content="", finish_reason="end_turn", event_type="done")
            return

        yield StreamChunk(id=req_id, model=model, delta_content="Live Claude Stream", event_type="token")
        yield StreamChunk(id=req_id, model=model, delta_content="", finish_reason="end_turn", event_type="done")
