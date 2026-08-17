from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.storage.database import db

router = APIRouter(prefix="/history", tags=["history"])

class SaveMessageRequest(BaseModel):
    id: str
    conversation_id: str
    sender: str
    content: str
    model_used: Optional[str] = None
    reasoning_steps: Optional[List[Any]] = Field(default_factory=list)
    sources: Optional[List[Any]] = Field(default_factory=list)

class CreateConversationRequest(BaseModel):
    id: str
    title: str
    model: Optional[str] = "aether-neural-local"

@router.get("/conversations")
async def get_conversations():
    """Retrieve list of all stored conversations."""
    return db.list_conversations()

@router.post("/conversations")
async def create_conversation(req: CreateConversationRequest):
    """Create or update conversation metadata."""
    db.create_or_update_conversation(req.id, req.title, req.model)
    return {"status": "success", "id": req.id}

@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """Retrieve full message history for a conversation."""
    return db.get_conversation_messages(conversation_id)

@router.post("/messages")
async def save_chat_message(req: SaveMessageRequest):
    """Save user or assistant message to database."""
    db.create_or_update_conversation(req.conversation_id, title=req.content[:30] if req.sender == "user" else "New Conversation")
    db.save_message(
        msg_id=req.id,
        conversation_id=req.conversation_id,
        sender=req.sender,
        content=req.content,
        model_used=req.model_used,
        reasoning_steps=req.reasoning_steps,
        sources=req.sources
    )
    return {"status": "saved", "id": req.id}

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete conversation and its message history."""
    db.delete_conversation(conversation_id)
    return {"status": "deleted", "id": conversation_id}
