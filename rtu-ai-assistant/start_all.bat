@echo off
title RTU AI Academic Assistant - Launch Unified Server
echo ==============================================================
echo RTU AI Academic Assistant: Unified Launch
echo ==============================================================
echo 1. Launching Unified Server and Academic Backend on http://127.0.0.1:8000 ...
start "RTU AI Assistant - Server" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

timeout /t 3 >nul

echo 2. Opening RTU Senior Tutor in default browser...
start http://127.0.0.1:8000/
