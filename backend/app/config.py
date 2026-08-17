import os
from dataclasses import dataclass

@dataclass
class Settings:
    PROJECT_NAME: str = "AETHER Proprietary AI Operating Layer"
    VERSION: str = "2.0.0-PROPRIETARY"
    API_V1_STR: str = "/api/v1"
    
    # Local ML & Neural Engine Configuration
    LOCAL_MODEL_NAME_OR_PATH: str = os.getenv("LOCAL_MODEL_NAME_OR_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    DEVICE: str = os.getenv("DEVICE", "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu")
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "1024"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    
    # Local Ollama / GGUF API endpoint (optional local host)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Search Engine Provider (Optional for real-time web search tool)
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    
    # Vector DB & RAG Storage
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "memory")  # 'memory', 'qdrant', 'pgvector'
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Defaults
    DEFAULT_MODEL: str = "aether-neural-local"

settings = Settings()

