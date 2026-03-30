@echo off
title JARVIS Voice Assistant
color 0A
echo.
echo     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
echo     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
echo     ██║███████║██████╔╝██║   ██║██║███████╗
echo ██  ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
echo ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
echo  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
echo.
echo   Voice Assistant — Powered by Porcupine + Whisper
echo   Say "Hey Jarvis" to activate
echo.
echo ─────────────────────────────────────────────────
echo.

cd /d "%~dp0"
python jarvis_core.py

if errorlevel 1 (
    echo.
    echo [ERROR] Jarvis crashed. Check logs\ folder for details.
    echo         Run setup.bat if you haven't installed dependencies.
    pause
)
