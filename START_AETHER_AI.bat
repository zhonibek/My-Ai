@echo off
chcp 437 >nul
title AETHER AI Platform Launcher
color 0B

echo.
echo ==============================================================================
echo          AETHER FRONTIER AI PLATFORM  ^|  100%% LOCAL NEURAL ENGINE
echo ==============================================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: Check Node
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found! Please install from https://nodejs.org
    pause
    exit /b 1
)

:: Set environment variables
set AETHER_SKIP_SENTENCE_TRANSFORMERS=0
set PYTHONIOENCODING=utf-8

echo [*] Starting Backend server (FastAPI + Uvicorn) on port 8000...
start "AETHER Backend" cmd /k "cd /d "%~dp0backend" && set AETHER_SKIP_SENTENCE_TRANSFORMERS=0 && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [*] Starting Frontend interface (Vite + React) on port 3000...
start "AETHER Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev -- --host"

echo.
echo [*] Waiting for servers to initialize (4 sec)...
timeout /t 4 /nobreak >nul

echo [*] Opening browser...
start http://localhost:3000

echo.
echo ==============================================================================
echo  [OK] Platform launched successfully!
echo.
echo   Frontend:  http://localhost:3000
echo   Backend:   http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo ==============================================================================
echo.
echo  Close the opened console windows to stop the platform.
echo.
pause
