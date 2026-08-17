from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', 'assistant', 'tool'")
    content: str = Field(..., description="Message text content")
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    context_window: int
    capabilities: List[str]  # e.g., ['chat', 'reasoning', 'vision', 'coding', 'tools']
    description: str
    is_default: bool = False

class CompletionResponse(BaseModel):
    id: str
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"
    tool_calls: Optional[List[Dict[str, Any]]] = None

class StreamChunk(BaseModel):
    id: str
    model: str
    delta_content: str
    finish_reason: Optional[str] = None
    tool_call_delta: Optional[Dict[str, Any]] = None
    event_type: str = "token"  # 'token', 'reasoning', 'tool_call', 'source_citation', 'done'
    metadata: Optional[Dict[str, Any]] = None

class ModelProvider(ABC):
    """Unified Provider-Agnostic Interface for LLM Models"""
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns provider identifier: 'openai', 'anthropic', 'google', 'vllm', etc."""
        pass

    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """List supported models for this provider"""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None
    ) -> CompletionResponse:
        """Non-streaming text completion generation"""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming text completion generation yielding StreamChunk objects"""
        pass


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source_domain: str
    published_date: Optional[str] = None

class SearchProvider(ABC):
    """Abstract Search Provider Interface for Web Search Abstraction"""
    
    @property
    @abstractmethod
    def provider_name(self) -> str: pass

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]: pass


class VectorChunk(BaseModel):
    id: str
    file_id: str
    file_name: str
    content: str
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None

class VectorDBProvider(ABC):
    """Abstract Vector Database Interface"""

    @abstractmethod
    async def insert_chunks(self, user_id: str, chunks: List[VectorChunk]) -> bool: pass

    @abstractmethod
    async def search_similar(self, user_id: str, query: str, top_k: int = 5, project_id: Optional[str] = None) -> List[VectorChunk]: pass


class EmbeddingProvider(ABC):
    """Abstract Embedding Provider Interface"""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]: pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]: pass
