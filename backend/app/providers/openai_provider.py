import json
import uuid
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional
from app.providers.base import ModelProvider, ChatMessage, CompletionResponse, StreamChunk, ModelInfo
from app.config import settings

class OpenAIProvider(ModelProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = "https://api.openai.com/v1"

    @property
    def provider_name(self) -> str:
        return "openai"

    async def list_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                id="gpt-4o",
                name="GPT-4o",
                provider="openai",
                context_window=128000,
                capabilities=["chat", "vision", "coding", "tools", "reasoning"],
                description="OpenAI flagship multimodal model for complex tasks",
                is_default=True
            ),
            ModelInfo(
                id="gpt-4o-mini",
                name="GPT-4o Mini",
                provider="openai",
                context_window=128000,
                capabilities=["chat", "coding", "tools", "fast"],
                description="Fast and economical model for everyday requests"
            ),
            ModelInfo(
                id="o1-mini",
                name="o1 Mini Reasoning",
                provider="openai",
                context_window=128000,
                capabilities=["chat", "reasoning", "coding", "math"],
                description="Advanced reasoning model specialized in STEM and logic"
            )
        ]

    async def generate(
        self,
        messages: List[ChatMessage],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None
    ) -> CompletionResponse:
        if not self.api_key:
            content = f"[Mock OpenAI Response ({model})]: I analyzed your query with {len(messages)} messages."
            return CompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                content=content,
                model=model,
                provider="openai",
                prompt_tokens=len(str(messages)) // 4,
                completion_tokens=len(content) // 4,
                finish_reason="stop"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [m.dict(exclude_none=True) for m in messages],
            "temperature": temperature
        }
        if tools:
            payload["tools"] = tools
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            
            choice = data["choices"][0]
            msg = choice["message"]
            usage = data.get("usage", {})
            
            return CompletionResponse(
                id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
                content=msg.get("content") or "",
                model=model,
                provider="openai",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                finish_reason=choice.get("finish_reason", "stop"),
                tool_calls=msg.get("tool_calls")
            )

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        if not self.api_key:
            mock_text = "This is a live streamed response from the **OpenAI Provider** (`" + str(model) + "`).\n\n- Model Gateway status: **Active**\n- Provider Agnostic Architecture: **Verified**\n\n```python\ndef solve_task(data):\n    return 'Processed ' + str(data)\n```"
            words = mock_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words)-1 else "")
                yield StreamChunk(
                    id=req_id,
                    model=model,
                    delta_content=chunk,
                    event_type="token"
                )
            yield StreamChunk(id=req_id, model=model, delta_content="", finish_reason="stop", event_type="done")
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [m.dict(exclude_none=True) for m in messages],
            "temperature": temperature,
            "stream": True
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            yield StreamChunk(id=req_id, model=model, delta_content="", finish_reason="stop", event_type="done")
                            break
                        try:
                            data = json.loads(data_str)
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            tool_calls = delta.get("tool_calls")
                            finish_reason = choice.get("finish_reason")

                            if content:
                                yield StreamChunk(
                                    id=data.get("id", req_id),
                                    model=model,
                                    delta_content=content,
                                    finish_reason=finish_reason,
                                    event_type="token"
                                )
                            elif tool_calls:
                                yield StreamChunk(
                                    id=data.get("id", req_id),
                                    model=model,
                                    delta_content="",
                                    tool_call_delta=tool_calls[0],
                                    event_type="tool_call"
                                )
                        except json.JSONDecodeError:
                            continue
