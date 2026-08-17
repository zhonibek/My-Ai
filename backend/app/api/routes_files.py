import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from pydantic import BaseModel

from app.rag.parser import doc_parser
from app.rag.chunker import text_chunker
from app.rag.vector_store import vector_store
from app.providers.base import VectorChunk

router = APIRouter(prefix="/files", tags=["files"])

class FileUploadResponse(BaseModel):
    file_id: str
    file_name: str
    size_bytes: int
    chunks_indexed: int
    status: str

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    user_id: str = Form("default_user")
):
    """
    Unified File & Image Upload Endpoint:
    Accepts both multipart/form-data (native browser drag & drop/file picker)
    and JSON payloads, extracts text/image details, chunks, and indexes into Vector Store.
    """
    filename = "uploaded_document.txt"
    content_bytes = b""

    # 1. Handle Multipart FormData Upload
    if file is not None:
        filename = file.filename or "uploaded_file"
        content_bytes = await file.read()
    else:
        # 2. Handle JSON Payload fallback
        try:
            body = await request.json()
            filename = body.get("file_name", "document.txt")
            raw_content = body.get("content", "")
            user_id = body.get("user_id", user_id)
            content_bytes = raw_content.encode("utf-8") if isinstance(raw_content, str) else b""
        except Exception:
            pass

    if not content_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file content is empty or invalid.")

    file_id = f"file_{uuid.uuid4().hex[:8]}"
    
    # Parse document or image content
    raw_text = doc_parser.parse_file(filename, content_bytes)
    
    # Generate chunks & store in vector store
    chunks = text_chunker.split_text(raw_text, file_id=file_id, file_name=filename)
    vector_chunks = [
        VectorChunk(
            id=c["chunk_id"],
            file_id=file_id,
            file_name=filename,
            content=c["content"],
            metadata={"chunk_index": c["chunk_index"]}
        )
        for c in chunks
    ]
    
    await vector_store.insert_chunks(user_id=user_id, chunks=vector_chunks)

    return FileUploadResponse(
        file_id=file_id,
        file_name=filename,
        size_bytes=len(content_bytes),
        chunks_indexed=len(vector_chunks),
        status="indexed"
    )
