import os
import sys
# Ensure app root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes_chat import router as chat_router
from app.api.routes_files import router as files_router
from app.api.routes_models import router as models_router
from app.api.routes_training import router as training_router
from app.api.routes_history import router as history_router
from app.api.routes_voice import router as voice_router
from app.api.routes_vision import router as vision_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Proprietary AI Model Engine & Operating Layer Platform Backend"
)

# Configure CORS for Next.js / Standalone frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(chat_router, prefix=settings.API_V1_STR)
app.include_router(files_router, prefix=settings.API_V1_STR)
app.include_router(models_router, prefix=settings.API_V1_STR)
app.include_router(training_router, prefix=settings.API_V1_STR)
app.include_router(history_router, prefix=settings.API_V1_STR)
app.include_router(voice_router, prefix=settings.API_V1_STR)
app.include_router(vision_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root_status():
    return {
        "status": "online",
        "platform": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "mode": "PROPRIETARY_MODEL_ENGINE",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
