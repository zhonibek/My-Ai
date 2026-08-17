import asyncio
import os
import sys

# Ensure backend root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from httpx import AsyncClient, ASGITransport
from app.main import app

async def test_root_endpoint():
    print("=== Testing FastAPI Root API Endpoint ===")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    print("Response JSON:", data)
    
    assert data["status"] == "online"
    assert "version" in data
    print("[SUCCESS] API Root Endpoint works correctly.")

if __name__ == "__main__":
    asyncio.run(test_root_endpoint())
