@echo off
title JARVIS Enhanced Command Manager
color 0A
echo.
echo     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
echo     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
echo     ██║███████║██████╔╝██║   ██║██║███████╗
echo ██  ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
echo ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
echo  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
echo.
echo   Enhanced Command Manager — Voice + Visual
echo   Features: Voice creation, testing, templates
echo.
echo ─────────────────────────────────────────────────
echo.

cd /d "%~dp0"
python jarvis_gui_enhanced.py

if errorlevel 1 (
    echo.
    echo [ERROR] Enhanced GUI crashed. Check dependencies:
    echo         • pip install tkinter (usually built-in)
    echo         • pip install pyaudio (for voice recording)
    echo         • OpenAI API key in config.json (for voice transcription)
    pause
)
