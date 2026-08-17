import os
import json
import time
import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional
from app.providers.base import ChatMessage, CompletionResponse, StreamChunk

class GGUFEngine:
    """
    High-Performance GGUF Quantized Model Engine.
    Supports direct loading of INT4/INT8/2-bit quantized GGUF model files
    via llama-cpp-python or llama.cpp subprocess for ultra-fast CPU inference.
    Ideal for deploying larger models (7B-14B) with minimal RAM usage.
    """
    def __init__(self):
        self.model = None
        self.model_path = None
        self._llama_available = False
        self._check_llama_cpp()

    def _check_llama_cpp(self):
        """Check if llama-cpp-python is installed."""
        try:
            from llama_cpp import Llama
            self._llama_available = True
        except ImportError:
            self._llama_available = False

    def load_model(self, model_path: str, n_ctx: int = 4096, n_threads: int = 8) -> bool:
        """
        Load a GGUF quantized model file.
        model_path: absolute path to a .gguf model file.
        """
        if not os.path.exists(model_path):
            print(f"[GGUF] Model file not found: {model_path}")
            return False

        if not self._llama_available:
            print("[GGUF] llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return False

        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False
            )
            self.model_path = model_path
            print(f"[GGUF] Successfully loaded: {os.path.basename(model_path)}")
            return True
        except Exception as e:
            print(f"[GGUF] Model load error: {e}")
            return False

    def is_ready(self) -> bool:
        return self.model is not None and self._llama_available

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Synchronous GGUF completion."""
        if not self.is_ready():
            return "[GGUF engine not loaded. Install llama-cpp-python and load a .gguf model.]"
        try:
            result = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False
            )
            return result["choices"][0]["text"].strip()
        except Exception as e:
            return f"[GGUF generation error: {e}]"

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        model: str = "gguf-local",
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async streaming interface compatible with the AETHER provider gateway."""
        import uuid

        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"<|system|>\n{msg.content}\n"
            elif msg.role == "user":
                prompt += f"<|user|>\n{msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"<|assistant|>\n{msg.content}\n"
        prompt += "<|assistant|>\n"

        req_id = f"gguf-{uuid.uuid4().hex[:8]}"

        if not self.is_ready():
            yield StreamChunk(
                id=req_id, model=model, delta_content="",
                event_type="reasoning",
                metadata={"status": "info", "message": "GGUF engine not loaded. Using fallback."}
            )
            return

        try:
            if self._llama_available:
                for chunk in self.model(prompt, max_tokens=max_tokens, temperature=temperature, stream=True, echo=False):
                    token = chunk["choices"][0].get("text", "")
                    if token:
                        yield StreamChunk(
                            id=req_id, model=model, delta_content=token, event_type="token"
                        )
        except Exception as e:
            yield StreamChunk(
                id=req_id, model=model,
                delta_content=f"[GGUF stream error: {e}]",
                event_type="token"
            )

    def get_status(self) -> Dict[str, Any]:
        return {
            "llama_cpp_available": self._llama_available,
            "model_loaded": self.is_ready(),
            "model_path": self.model_path or "No model loaded",
            "instructions": "Place a .gguf file in backend/data/models/ and call /api/v1/models/load_gguf"
        }

gguf_engine = GGUFEngine()
