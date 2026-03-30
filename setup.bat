@echo off
echo ============================================================
echo   JARVIS Voice Assistant — Setup
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo [1/2] Installing Python dependencies...
pip install openwakeword pyaudio openai numpy onnxruntime --upgrade

if errorlevel 1 (
    echo.
    echo [NOTE] If PyAudio fails, try:
    echo   pip install pipwin
    echo   pipwin install pyaudio
    echo.
)

echo.
echo [2/2] Downloading wake-word models (one-time, ~5 MB)...
python -c "import openwakeword; openwakeword.utils.download_models(); print('Models downloaded!')"

echo.
echo ============================================================
echo   Setup complete!  NO API keys needed for wake word.
echo   (Only your OpenAI key in config.json for Whisper STT)
echo.
echo   Run Jarvis with:
echo     python jarvis_core.py
echo   Or double-click START_JARVIS.bat
echo ============================================================
pause
