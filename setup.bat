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

echo [1/3] Installing Python dependencies...
pip install pvporcupine pyaudio openai --upgrade

if errorlevel 1 (
    echo.
    echo [NOTE] If PyAudio fails, try:
    echo   pip install pipwin
    echo   pipwin install pyaudio
    echo.
)

echo.
echo [2/3] Getting Porcupine Access Key...
echo.
echo   Porcupine requires a FREE access key from Picovoice.
echo   1. Go to: https://console.picovoice.ai/
echo   2. Sign up (free)
echo   3. Copy your Access Key
echo.
set /p ACCESS_KEY="  Paste your Access Key here: "

if not "%ACCESS_KEY%"=="" (
    setx PORCUPINE_ACCESS_KEY "%ACCESS_KEY%"
    set PORCUPINE_ACCESS_KEY=%ACCESS_KEY%
    echo   ✓ Access Key saved to environment
) else (
    echo   [SKIP] No key entered — set PORCUPINE_ACCESS_KEY later
)

echo.
echo [3/3] Verifying installation...
python -c "import pvporcupine; import pyaudio; import openai; print('All packages OK!')"

echo.
echo ============================================================
echo   Setup complete! Run Jarvis with:
echo     python jarvis_core.py
echo   Or double-click START_JARVIS.bat
echo ============================================================
pause
