from fastapi import APIRouter, HTTPException, UploadFile, File, Response
from pydantic import BaseModel
from app.voice.voice_engine import voice_engine

router = APIRouter(prefix="/voice", tags=["voice"])

class SynthesizeRequest(BaseModel):
    text: str
    lang: str = "ru"

@router.post("/synthesize")
async def synthesize_voice(req: SynthesizeRequest):
    """
    Synthesize text into streaming audio bytes (MP3 / WAV).
    """
    if not req.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    audio_bytes = await voice_engine.synthesize_speech(req.text, lang=req.lang)
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"}
    )

@router.post("/transcribe")
async def transcribe_voice(file: UploadFile = File(...)):
    """
    Transcribe uploaded microphone audio recording into text.
    """
    audio_data = await file.read()
    text = await voice_engine.transcribe_audio(audio_data, filename=file.filename)
    return {"text": text, "status": "success"}
