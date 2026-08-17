from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from app.vision.vision_engine import vision_engine

router = APIRouter(prefix="/vision", tags=["vision"])

@router.post("/analyze")
async def analyze_image(
    prompt: str = Form(default="Describe this image in detail."),
    file: UploadFile = File(...)
):
    """
    Multimodal image analysis: upload a screenshot, diagram, chart or code image,
    receive structured visual context grounded into LLM-ready description.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    result = vision_engine.analyze_image_bytes(image_bytes, prompt=prompt)

    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("message"))

    return result
