import json
import uuid
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional
from app.providers.base import ModelProvider, ChatMessage, CompletionResponse, StreamChunk, ModelInfo
from app.config import settings


class KimiK3Provider(ModelProvider):
    """
    Kimi-K3 (MoonshotAI) — OpenAI-compatible provider.
    Base URL: https://api.moonshot.ai/v1
    Model ID: kimi-k3
    Supports: 1M context, thinking/reasoning, streaming
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.KIMI_API_KEY
        self.base_url = "https://api.moonshot.ai/v1"
        self.default_model = "kimi-k3"

    @property
    def provider_name(self) -> str:
        return "kimi"

    async def list_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                id="kimi-k3",
                name="Kimi-K3 (MoonshotAI)",
                provider="kimi",
                context_window=1_000_000,
                capabilities=["chat", "coding", "reasoning", "math", "analysis", "1M context", "thinking"],
                description="Kimi K3 — frontier MoE model with 1M token context and always-on chain-of-thought reasoning",
                is_default=False
            ),
        ]

    async def generate(
        self,
        messages: List[ChatMessage],
        model: str = "kimi-k3",
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> CompletionResponse:
        req_id = f"kimi-{uuid.uuid4().hex[:8]}"

        if not self.api_key:
            return CompletionResponse(
                id=req_id,
                content=(
                    "⚠️ **Kimi-K3 не подключён**: API-ключ не задан.\n\n"
                    "Добавьте в файл `.env` (в папке `backend/`):\n"
                    "```\nKIMI_API_KEY=ваш_ключ_с_platform.kimi.ai\n```\n"
                    "Получить ключ: https://platform.kimi.ai"
                ),
                model=model,
                provider="kimi",
                prompt_tokens=0,
                completion_tokens=0,
                finish_reason="stop",
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [m.dict(exclude_none=True) for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            res.raise_for_status()
            data = res.json()

        choice = data["choices"][0]
        msg = choice["message"]
        usage = data.get("usage", {})

        # Kimi-K3 может возвращать reasoning_content отдельно
        reasoning = msg.get("reasoning_content", "")
        content = msg.get("content", "") or ""
        if reasoning:
            content = f"<think>\n{reasoning}\n</think>\n\n{content}"

        return CompletionResponse(
            id=data.get("id", req_id),
            content=content,
            model=model,
            provider="kimi",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        model: str = "kimi-k3",
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        req_id = f"kimi-{uuid.uuid4().hex[:8]}"

        if not self.api_key:
            no_key_msg = (
                "⚠️ **Kimi-K3 не подключён**: API-ключ не задан. "
                "Добавьте `KIMI_API_KEY=ваш_ключ` в файл `backend/.env`. "
                "Получить: https://platform.kimi.ai"
            )
            words = no_key_msg.split(" ")
            for i, word in enumerate(words):
                yield StreamChunk(
                    id=req_id, model=model,
                    delta_content=word + (" " if i < len(words) - 1 else ""),
                    event_type="token",
                )
            yield StreamChunk(id=req_id, model=model, delta_content="", finish_reason="stop", event_type="done")
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [m.dict(exclude_none=True) for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools

        in_thinking = False
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        yield StreamChunk(
                            id=req_id, model=model, delta_content="",
                            finish_reason="stop", event_type="done",
                        )
                        return
                    try:
                        data = json.loads(data_str)
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})

                        # reasoning_content (thinking) — Kimi-K3 особенность
                        reasoning_delta = delta.get("reasoning_content", "")
                        content_delta = delta.get("content", "")
                        finish_reason = choice.get("finish_reason")

                        if reasoning_delta:
                            if not in_thinking:
                                yield StreamChunk(id=req_id, model=model, delta_content="<think>\n", event_type="token")
                                in_thinking = True
                            yield StreamChunk(id=req_id, model=model, delta_content=reasoning_delta, event_type="token")

                        if content_delta:
                            if in_thinking:
                                yield StreamChunk(id=req_id, model=model, delta_content="\n</think>\n\n", event_type="token")
                                in_thinking = False
                            yield StreamChunk(
                                id=data.get("id", req_id),
                                model=model,
                                delta_content=content_delta,
                                finish_reason=finish_reason,
                                event_type="token",
                            )

                        if finish_reason and finish_reason != "null":
                            if in_thinking:
                                yield StreamChunk(id=req_id, model=model, delta_content="\n</think>\n", event_type="token")
                            yield StreamChunk(
                                id=req_id, model=model, delta_content="",
                                finish_reason=finish_reason, event_type="done",
                            )
                            return

                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


kimi_provider = KimiK3Provider()
