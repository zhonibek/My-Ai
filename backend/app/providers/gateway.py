from typing import List, Dict, Tuple
from app.providers.base import ModelProvider, ModelInfo
from app.inference.local_engine import proprietary_engine


class ModelGateway:
    """
    Gateway registry managing 100% Local Proprietary AI Engines:
    - AETHER Neural Engine (Local Transformer RoPE-SwiGLU)
    - AETHER Research Models (ai-research PyTorch architectures)
    - Ollama Local (DeepSeek-R1 / Qwen2.5 / Llama3.2 offline)
    """

    def __init__(self):
        self.providers: Dict[str, ModelProvider] = {
            "proprietary": proprietary_engine,
        }

    async def list_all_models(self) -> List[ModelInfo]:
        all_models = []
        for provider in self.providers.values():
            try:
                models = await provider.list_models()
                all_models.extend(models)
            except Exception:
                continue
        return all_models

    def resolve_provider(self, model_id: str) -> Tuple[ModelProvider, str]:
        """
        Routes all model requests to the Local Proprietary Engine.
        """
        return self.providers["proprietary"], model_id


gateway = ModelGateway()

