@echo off
title RTU AI Academic Assistant - Backend Service
echo ==============================================================
echo Starting RTU AI Academic Assistant FastAPI Backend Server...
echo ==============================================================
cd /d "%~dp0backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause

