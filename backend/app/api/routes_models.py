from typing import List
from fastapi import APIRouter
from app.providers.base import ModelInfo
from app.providers.gateway import gateway

router = APIRouter(prefix="/models", tags=["models"])

@router.get("", response_model=List[ModelInfo])
async def get_models_list():
    """
    List all available LLM models across OpenAI, Anthropic, Google, and Self-Hosted endpoints.
    """
    models = await gateway.list_all_models()
    return models
