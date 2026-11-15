@echo off
title RTU AI Academic Assistant - Unified Launcher
echo ==============================================================
echo RTU AI Academic Assistant: Starting Backend + Frontend
echo ==============================================================
cd /d "%~dp0"

echo 1. Starting FastAPI Backend Server on http://127.0.0.1:8000 ...
start "RTU AI Backend" cmd /k "cd /d "%~dp0rtu-ai-assistant\backend" && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

echo 2. Waiting for backend to initialize FAISS and database...
timeout /t 3 >nul

echo 3. Starting Streamlit UI...
start "RTU AI Streamlit UI" cmd /k "cd /d "%~dp0" && streamlit run streamlit_app.py"

echo All services launched.
