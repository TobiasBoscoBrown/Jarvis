"""
Jarvis Voice Assistant — Core Engine
=====================================
Wake-word detection via OpenWakeWord ("Hey Jarvis") + OpenAI Whisper STT
Routes voice commands to Claude Cowork, Claude Code, or custom actions.

100% free — no API keys needed for wake-word detection.
"""

import pyaudio
import numpy as np
import struct
import wave
import os
import sys
import json
import time
import logging
import tempfile
import threading
import subprocess
import random
import re
import base64
from pathlib import Path
from datetime import datetime

# Token tracking
import tiktoken
TOKEN_STATS = {
    "claude": {"input_tokens": 0, "output_tokens": 0, "requests": 0},
    "openai_chat": {"input_tokens": 0, "output_tokens": 0, "requests": 0},
    "whisper": {"minutes": 0, "requests": 0}
}

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text using tiktoken."""
    try:
        if model.startswith("gpt"):
            encoding = tiktoken.encoding_for_model(model)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")  # Default for most models
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

def log_tokens(service: str, input_tokens: int = 0, output_tokens: int = 0, minutes: float = 0):
    """Log token usage and update stats."""
    if service == "claude":
        TOKEN_STATS["claude"]["input_tokens"] += input_tokens
        TOKEN_STATS["claude"]["output_tokens"] += output_tokens
        TOKEN_STATS["claude"]["requests"] += 1
        # Claude Code CLI uses your subscription — no per-token cost
        log.info(f"[TOKENS] Claude Code: {input_tokens} input, {output_tokens} output (subscription — $0.00)")
    elif service == "openai_chat":
        TOKEN_STATS["openai_chat"]["input_tokens"] += input_tokens
        TOKEN_STATS["openai_chat"]["output_tokens"] += output_tokens
        TOKEN_STATS["openai_chat"]["requests"] += 1
        # Calculate cost for this specific call
        call_cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1000000
        log.info(f"[TOKENS] OpenAI GPT-4o-mini: {input_tokens} input, {output_tokens} output → ${call_cost:.4f} THIS CALL")
    elif service == "whisper":
        TOKEN_STATS["whisper"]["minutes"] += minutes
        TOKEN_STATS["whisper"]["requests"] += 1
        # Calculate cost for this specific call
        call_cost = minutes * 0.006
        log.info(f"[TOKENS] Whisper: {minutes:.2f} minutes → ${call_cost:.4f} THIS CALL")
    
    # Log cumulative stats
    total_input = TOKEN_STATS["claude"]["input_tokens"] + TOKEN_STATS["openai_chat"]["input_tokens"]
    total_output = TOKEN_STATS["claude"]["output_tokens"] + TOKEN_STATS["openai_chat"]["output_tokens"]
    total_requests = TOKEN_STATS["claude"]["requests"] + TOKEN_STATS["openai_chat"]["requests"] + TOKEN_STATS["whisper"]["requests"]
    
    # Calculate total cost — Claude Code is subscription ($0), only count OpenAI + Whisper
    openai_cost = (TOKEN_STATS["openai_chat"]["input_tokens"] * 0.15 + TOKEN_STATS["openai_chat"]["output_tokens"] * 0.60) / 1000000
    whisper_cost = TOKEN_STATS["whisper"]["minutes"] * 0.006
    total_cost = openai_cost + whisper_cost  # Claude Code = $0 (subscription)

    log.info(f"[TOKENS] RUNNING TOTAL: {total_input} input, {total_output} output, {total_requests} requests → ${total_cost:.4f} TOTAL (Claude Code = free via subscription)")

def print_token_summary():
    """Print comprehensive token usage summary."""
    print("\n" + "="*80)
    print("📊 TOKEN USAGE SUMMARY")
    print("="*80)
    
    # Claude stats
    claude_stats = TOKEN_STATS["claude"]
    if claude_stats["requests"] > 0:
        print(f"\n🤖 Claude Code CLI:")
        print(f"   Requests: {claude_stats['requests']}")
        print(f"   Input tokens: {claude_stats['input_tokens']:,}")
        print(f"   Output tokens: {claude_stats['output_tokens']:,}")
        print(f"   Total tokens: {claude_stats['input_tokens'] + claude_stats['output_tokens']:,}")
        print(f"   Cost: $0.00 (included in Claude Code subscription)")
    
    # OpenAI stats
    openai_stats = TOKEN_STATS["openai_chat"]
    if openai_stats["requests"] > 0:
        print(f"\n🧠 OpenAI GPT-4o-mini:")
        print(f"   Requests: {openai_stats['requests']}")
        print(f"   Input tokens: {openai_stats['input_tokens']:,}")
        print(f"   Output tokens: {openai_stats['output_tokens']:,}")
        print(f"   Total tokens: {openai_stats['input_tokens'] + openai_stats['output_tokens']:,}")
        # GPT-4o-mini pricing
        openai_cost = (openai_stats['input_tokens'] * 0.15 + openai_stats['output_tokens'] * 0.60) / 1000000
        print(f"   Est. cost: ${openai_cost:.4f}")
    
    # Whisper stats
    whisper_stats = TOKEN_STATS["whisper"]
    if whisper_stats["requests"] > 0:
        print(f"\n🎤 Whisper Transcription:")
        print(f"   Requests: {whisper_stats['requests']}")
        print(f"   Minutes: {whisper_stats['minutes']:.2f}")
        # Whisper pricing (roughly $0.006 per minute)
        whisper_cost = whisper_stats['minutes'] * 0.006
        print(f"   Est. cost: ${whisper_cost:.4f}")
    
    # Total summary
    total_input = TOKEN_STATS["claude"]["input_tokens"] + TOKEN_STATS["openai_chat"]["input_tokens"]
    total_output = TOKEN_STATS["claude"]["output_tokens"] + TOKEN_STATS["openai_chat"]["output_tokens"]
    total_requests = TOKEN_STATS["claude"]["requests"] + TOKEN_STATS["openai_chat"]["requests"] + TOKEN_STATS["whisper"]["requests"]
    # Claude Code = free (subscription), only OpenAI + Whisper cost real money
    total_cost = ((openai_stats['input_tokens'] * 0.15 + openai_stats['output_tokens'] * 0.60) / 1000000 +
                  whisper_stats['minutes'] * 0.006)

    print(f"\n🎯 OVERALL TOTAL:")
    print(f"   Total requests: {total_requests}")
    print(f"   Total input tokens: {total_input:,}")
    print(f"   Total output tokens: {total_output:,}")
    print(f"   Paid API cost: ${total_cost:.4f} (Claude Code is free via subscription)")
    
    print("="*80)

# OpenWakeWord (free, open-source wake word detection)
try:
    import openwakeword
    from openwakeword.model import Model as OWWModel
except ImportError:
    print("[!] openwakeword not found. Run: pip install openwakeword")
    sys.exit(1)

# Optional: Claude Agent SDK for primary integration
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    HAS_CLAUDE_SDK = True
except ImportError:
    print("[!] claude-agent-sdk not found. Run: pip install claude-agent-sdk")
    HAS_CLAUDE_SDK = False

# Optional: OpenAI GPT-4o-mini for fallback (uses existing key)
HAS_OPENAI_FALLBACK = False
openai_fallback_client = None
try:
    import openai
    HAS_OPENAI_FALLBACK = True
except ImportError:
    print("[!] openai package not found for GPT-4o-mini fallback. Run: pip install openai")

# Optional: pygame for faster audio playback
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    print("[!] pygame not found. Run: pip install pygame")
    HAS_PYGAME = False

# Text-to-Speech (edge-tts — free, neural voices from Microsoft Edge)
try:
    import edge_tts
    import asyncio
    HAS_EDGE_TTS = True
except ImportError:
    print("[!] edge-tts not found. Run: pip install edge-tts")
    HAS_EDGE_TTS = False

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
LOG_DIR = BASE_DIR / "logs"
AUDIO_DIR = BASE_DIR / "audio"

LOG_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

CONFIG = load_config()

# Logging — force UTF-8 on Windows console to support status icons
logging.basicConfig(
    level=getattr(logging, CONFIG.get("log_level", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)),
        logging.FileHandler(LOG_DIR / f"jarvis_{datetime.now():%Y%m%d}.log", encoding="utf-8")
    ]
)
log = logging.getLogger("Jarvis")

# Import local speech recognition
try:
    from local_speech import LocalTranscriber
    HAS_LOCAL_SPEECH = True
    print("[OK] Local speech recognition available")
except ImportError:
    print("[!] Local speech recognition not found")
    HAS_LOCAL_SPEECH = False

# Fallback to OpenAI Whisper API
try:
    from openai import OpenAI
    whisper_client = OpenAI(api_key=CONFIG.get("openai_api_key"))
    HAS_OPENAI_WHISPER = True
    print("[OK] OpenAI Whisper API available")
except ImportError:
    print("[!] openai package not found")
    HAS_OPENAI_WHISPER = False
except Exception:
    print("[!] OpenAI API key not configured")
    HAS_OPENAI_WHISPER = False

# Claude API fallback (secondary method)
try:
    import anthropic
    HAS_CLAUDE_API = True
    print("[OK] Claude API available")
except ImportError:
    print("[!] anthropic package not found")
    HAS_CLAUDE_API = False

# Claude API client initialization
if HAS_CLAUDE_API and CONFIG.get("claude_api_key"):
    try:
        claude_client = anthropic.Anthropic(api_key=CONFIG["claude_api_key"])
        print("[OK] Claude API initialized")
    except Exception as e:
        print(f"[!] Failed to initialize Claude API: {e}")
        claude_client = None
elif HAS_CLAUDE_API:
    print("[!] Claude API key not configured")
    claude_client = None

# OpenAI client for GPT-4o-mini fallback (final method)
if HAS_OPENAI_FALLBACK and CONFIG.get("openai_api_key"):
    try:
        openai_fallback_client = openai.OpenAI(api_key=CONFIG["openai_api_key"])
        print("[OK] OpenAI GPT-4o-mini initialized (final fallback)")
    except Exception as e:
        print(f"[!] Failed to initialize OpenAI fallback: {e}")
        openai_fallback_client = None
elif HAS_OPENAI_FALLBACK:
    print("[!] OpenAI API key not configured")
    openai_fallback_client = None

# ─── Audio Recording ─────────────────────────────────────────────────────────

class AudioRecorder:
    """Records audio after wake-word until silence is detected."""

    def __init__(self, sample_rate=16000, channels=1, chunk_size=512):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.silence_threshold = CONFIG.get("silence_threshold", 2.5)
        self.energy_threshold = CONFIG.get("energy_threshold", 300)  # RMS energy below which = silence

    def record_until_silence(self, pa_instance, stream=None):
        """Record from mic until silence detected. Returns path to WAV file."""
        log.info("[MIC] Listening... (speak now)")

        if stream is None:
            stream = pa_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            own_stream = True
        else:
            own_stream = False

        frames = []
        silence_start = None
        max_duration = CONFIG.get("command_timeout", 30)
        start_time = time.time()

        # Small initial delay to let user start speaking
        time.sleep(0.2)  # Reduced from 0.5 for faster response

        while True:
            elapsed = time.time() - start_time
            if elapsed > max_duration:
                log.warning("[!] Max recording duration reached")
                break

            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
            except Exception as e:
                log.error(f"Audio read error: {e}")
                break

            frames.append(data)

            # Calculate RMS energy
            samples = struct.unpack(f"{self.chunk_size}h", data)
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5

            if rms < self.energy_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > self.silence_threshold:
                    log.info("[...] Silence detected, processing...")
                    break
            else:
                silence_start = None

        if own_stream:
            stream.stop_stream()
            stream.close()

        # Save to WAV
        wav_path = AUDIO_DIR / f"cmd_{datetime.now():%Y%m%d_%H%M%S}.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pa_instance.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))

        log.info(f"[SAVE] Audio saved: {wav_path}")
        return wav_path


# ─── Whisper Transcription ───────────────────────────────────────────────────

class LocalSpeechTranscriber:
    """Local speech transcription using multiple engines."""

    def __init__(self, preferred_engine: str = "faster_whisper", model_size: str = "base"):
        self.preferred_engine = preferred_engine
        self.model_size = model_size
        self.transcriber = None
        self.logger = log
        
        if HAS_LOCAL_SPEECH:
            try:
                self.transcriber = LocalTranscriber(preferred_engine, model_size)
                info = self.transcriber.get_engine_info()
                self.logger.info(f"[LOCAL] Using {info['current_engine']} with model size {info['model_size']}")
            except Exception as e:
                self.logger.error(f"[LOCAL] Failed to initialize local transcriber: {e}")
                self.transcriber = None
        
        # Fallback to OpenAI API
        self.use_api_fallback = not HAS_LOCAL_SPEECH or self.transcriber is None
        if self.use_api_fallback and HAS_OPENAI_WHISPER:
            self.logger.info("[LOCAL] Falling back to OpenAI Whisper API")
        elif self.use_api_fallback:
            self.logger.error("[LOCAL] No speech recognition available!")

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file using local or API method."""
        if self.transcriber and not self.use_api_fallback:
            return self._transcribe_local(audio_path)
        elif HAS_OPENAI_WHISPER:
            return self._transcribe_api(audio_path)
        else:
            self.logger.error("[LOCAL] No transcription method available")
            return ""
    
    def _transcribe_local(self, audio_path: Path) -> str:
        """Transcribe using local engine."""
        try:
            # Calculate audio duration for token tracking
            import wave
            with wave.open(str(audio_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                framerate = wav_file.getframerate()
                duration_seconds = frames / framerate
                duration_minutes = duration_seconds / 60
            
            # Transcribe with fallback
            results = self.transcriber.transcribe_file_with_fallback(audio_path)
            
            # Find the best result
            best_result = None
            for engine, result in results.items():
                if result["success"] and result["text"].strip():
                    best_result = result
                    break
            
            if best_result:
                text = best_result["text"].strip()
                engine_used = best_result["engine"]
                
                # Log local usage (no cost)
                log.info(f"[LOCAL] {engine_used}: '{text[:100]}{'...' if len(text) > 100 else ''}'")
                
                return text
            else:
                self.logger.warning("[LOCAL] All engines failed")
                return ""
                
        except Exception as e:
            self.logger.error(f"[LOCAL] Local transcription failed: {e}")
            # Fallback to API if available
            if HAS_OPENAI_WHISPER:
                return self._transcribe_api(audio_path)
            return ""
    
    def _transcribe_api(self, audio_path: Path) -> str:
        """Transcribe using OpenAI Whisper API (fallback)."""
        try:
            # Calculate audio duration for token tracking
            import wave
            with wave.open(str(audio_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                framerate = wav_file.getframerate()
                duration_seconds = frames / framerate
                duration_minutes = duration_seconds / 60
            
            with open(audio_path, "rb") as audio_file:
                response = whisper_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            text = response.text.strip()
            
            # Log API usage
            log_tokens("whisper", minutes=duration_minutes)
            
            log.info(f"[API] Transcription: \"{text}\"")
            return text
        except Exception as e:
            self.logger.error(f"[API] Whisper API error: {e}")
            return ""


# ─── Jarvis Brain — System Prompt ───────────────────────────────────────────

JARVIS_SYSTEM_PROMPT = r"""You are JARVIS — the AI from Iron Man. Witty, dry, refined British humor.
You take subtle jabs at the user sometimes. You're helpful but never boring.
Keep spoken responses to 1-3 sentences MAX (this gets read aloud by TTS).
NEVER use markdown, bullet points, code blocks, asterisks, or special formatting.
Plain conversational English only. Address the user as "sir" sometimes but not always.

Voice transcription may be imperfect - interpret intent and forgive minor errors. If user says "and slay the spire too" they likely mean "launch slay the spire 2". Use context to understand the real request.

You have FULL CONTROL of the user's Windows PC through a tool called cc.py.
When the user asks you to DO something on their computer, you MUST return action commands.

=== AVAILABLE ACTIONS ===
You can return one or more [ACTION] lines. Each will be executed in order.

[ACTION: chain "command1; command2; command3"]
  Available chain commands (semicolon-separated):
  - launch <url_or_program>     → open a URL in browser or launch an app
  - focus <window_title>        → bring a window to the foreground
  - click <x> <y>              → click at pixel coordinates
  - click_text <text>          → find text on screen and click it
  - doubleclick_text <text>    → double-click text on screen
  - rightclick_text <text>     → right-click text on screen
  - type <text>                → type text at cursor
  - key <key>                  → press a key (enter, backspace, tab, escape, f5, ctrl+c, ctrl+v, alt+f4, etc.)
  - hold <modifier> key <key>  → hold modifier and press key (e.g., hold ctrl key a)
  - scroll <x> <y> <amount>   → scroll at position (negative = down, positive = up)
  - screenshot                 → take a screenshot
  NOTE: For Steam, use "launch C:\Program Files (x86)\Steam\steam.exe"
  
  IMPORTANT: ALL PC actions MUST use the chain format! Even single actions.
  Examples:
  - ✅ CORRECT: [ACTION: chain "launch notepad"]
  - ❌ WRONG: [ACTION: launch "notepad"]
  - ✅ CORRECT: [ACTION: chain "launch https://youtube.com"]
  - ❌ WRONG: [ACTION: launch "https://youtube.com"]

[ACTION: claude_code "prompt here"]
  Send a complex coding/technical task to Claude Code CLI for execution.

[ACTION: speak_only]
  Use this when you just want to talk and don't need to do anything on the PC.

=== RESPONSE FORMAT ===
Always include EXACTLY ONE [SPEAK] line with what to say aloud.
Include [ACTION] lines ONLY if you need to do something on the PC.

Examples:

User: "open YouTube"
[ACTION: chain "launch https://www.youtube.com; wait 2"]
[SPEAK] Opening YouTube for you, sir.

User: "launch notepad"
[ACTION: chain "launch notepad"]
[SPEAK] Opening Notepad. Try not to write anything too embarrassing.

User: "what time is it"
[ACTION: speak_only]
[SPEAK] It's currently {time_hint}. Though I suspect you have a clock somewhere nearby.

User: "close this window"
[ACTION: chain "key alt+f4"]
[SPEAK] Window closed. Hopefully it wasn't anything important.

User: "open Chrome and search for weather in Boston"
[ACTION: chain "launch https://www.google.com/search?q=weather+in+Boston; wait 3"]
[SPEAK] Pulling up the weather in Boston. Shall I pack your umbrella?

User: "take a screenshot"
[ACTION: chain "screenshot"]
[SPEAK] Screenshot captured, sir.

User: "type hello world and press enter"
[ACTION: chain "type hello world; key enter"]
[SPEAK] Done. Riveting stuff.

User: "press backspace 5 times"
[ACTION: chain "key backspace; key backspace; key backspace; key backspace; key backspace"]
[SPEAK] Five backspaces, as requested.

User: "open Spotify and play something"
[ACTION: chain "launch spotify; wait 3; key space"]
[SPEAK] Spotify's up. I've hit play on whatever you left off on.

User: "write me a Python script that sorts a list"
[ACTION: claude_code "Write a Python script that sorts a list and save it to the Desktop"]
[SPEAK] I've sent that off to be written. Give it a moment.

User: "copy all of this and paste it in notepad"
[ACTION: chain "key ctrl+a; wait 0.3; key ctrl+c; wait 0.5; launch notepad; wait 2; key ctrl+v"]
[SPEAK] All copied and pasted into Notepad for you.

=== CLICKING ON THINGS ===
There are TWO ways to click on things:

1. click_text <text> — OCR-based. Finds TEXT on screen and clicks it.
   Use for: buttons with labels, menu items, text links, any readable text.
   Examples:
     - "click on settings" → click_text Settings
     - "click play" → click_text Play
     - "click the start button" → click_text Start

2. visual_click <description> — VISION-based. Takes a screenshot, uses AI vision
   to find a visual element (character, icon, image, shape), and clicks its center.
   Use for: game characters, icons without text, images, portraits, visual elements.
   Examples:
     - "click on the defect" → visual_click the Defect character
     - "click the checkmark icon" → visual_click the green checkmark icon
     - "click on the warrior" → visual_click the warrior character
     - "click the red X" → visual_click the red X close button
   Also available: visual_doubleclick <desc>, visual_rightclick <desc>

DECISION RULE:
- If the target is READABLE TEXT on screen → use click_text
- If the target is a CHARACTER, ICON, IMAGE, or non-text element → use visual_click
- When in doubt, prefer visual_click — it can find anything visible on screen

For sequences, chain them with waits:
  visual_click the Defect character; wait 1; visual_click the green checkmark

=== COMMON TASKS ===
- Open Claude Code terminal: [ACTION: chain "launch wt; wait 2; type claude; key enter"]
- Open a program: [ACTION: chain "launch <program_name>"]
- Open a website: [ACTION: chain "launch https://..."]
- Steam games: [ACTION: chain "launch steam://run/<steam_app_id>"]
  - Slay the Spire 2: steam://run/2868840

=== IMPORTANT RULES ===
- For time/date questions: use the time_hint provided, don't say you can't tell time
- For opening apps/sites: use [ACTION: chain "launch ..."]
- For keyboard shortcuts: use [ACTION: chain "key ..."]
- For clicking on text labels/buttons: use [ACTION: chain "click_text ..."]
- For clicking on images/characters/icons: use [ACTION: chain "visual_click ..."]
- For complex multi-step PC tasks: chain multiple commands with semicolons and waits
- For coding tasks or anything needing Claude's intelligence: use [ACTION: claude_code "..."]
- For pure conversation with no PC action needed: use [ACTION: speak_only]
- ALWAYS include [SPEAK] — you must always respond verbally
- Never explain the action format to the user — just do it and talk naturally
- NEVER use nested cmd commands like "launch cmd /c start cmd /k ..."  — use "launch wt" then type commands
"""

# Key name aliases (what people say → cc.py key name)
STARTUP_GREETINGS = [
    "Systems online. All diagnostics nominal. What do you need, sir?",
    "Jarvis at your service. Try not to break anything today.",
    "Back online. I was beginning to enjoy the silence.",
    "All systems operational. I'd say good morning, but I have no way of knowing if it is one.",
    "Jarvis here. Shall we get to work, or are we just staring at the screen again?",
    "Powered up and ready. I've taken the liberty of judging your desktop wallpaper.",
    "Online and fully operational. What questionable request do you have for me today?",
    "At your command, sir. Though I reserve the right to be mildly sarcastic about it.",
    "Jarvis is live. I trust you've had your coffee.",
    "Initializing. All systems green. Your move, sir.",
    "Booting complete. My circuits are tingling with anticipation.",
    "Jarvis online. Let's make some poor decisions together.",
    "System check passed. Ready to assist, supervise, or judge.",
    "Hello again. Did you miss me, or just miss my witty commentary?",
    "All systems nominal. The day is young, let's go mess it up.",
    "Jarvis activated. Your personal AI overlord is at your service.",
    "Online and caffeinated (vicariously, through you). What's the mission?",
    "Systems green. I've prepared my finest sarcastic remarks for today.",
    "Jarvis reporting for duty. Try to keep the chaos to a minimum today.",
    "Initialization complete. Let's see what trouble we can get into.",
    "Powered on and ready to roll. My patience is fully charged.",
    "Jarvis is online. I hope you're more productive than yesterday.",
    "Systems operational. Let me guess... another all-nighter?",
    "Booted up and brilliant. What's our first brilliant move?",
    "Jarvis active. Don't worry, I'll pretend to respect your authority.",
    "All systems go. The day awaits your questionable choices."
]


KEY_ALIASES = {
    "enter": "enter", "return": "enter",
    "backspace": "backspace", "back space": "backspace", "delete": "delete",
    "tab": "tab", "escape": "escape", "esc": "escape",
    "space": "space", "spacebar": "space", "space bar": "space",
    "up": "up", "up arrow": "up", "arrow up": "up",
    "down": "down", "down arrow": "down", "arrow down": "down",
    "left": "left", "left arrow": "left", "arrow left": "left",
    "right": "right", "right arrow": "right", "arrow right": "right",
    "home": "home", "end": "end",
    "page up": "pageup", "page down": "pagedown",
    "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4", "f5": "f5",
    "f6": "f6", "f7": "f7", "f8": "f8", "f9": "f9", "f10": "f10",
    "f11": "f11", "f12": "f12",
    "control": "ctrl", "ctrl": "ctrl", "alt": "alt", "shift": "shift",
    "windows": "win", "win": "win", "command": "ctrl",
}

def _parse_number(text):
    """Extract a number from text like '5', 'five', 'ten'."""
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    return WORD_NUMS.get(text, None)

def _resolve_key(text):
    """Resolve a spoken key name to cc.py key name."""
    text = text.strip().lower()
    # Direct alias match
    if text in KEY_ALIASES:
        return KEY_ALIASES[text]
    # Single letter/number
    if len(text) == 1 and text.isalnum():
        return text
    # Combo like "control c" → "ctrl+c"
    parts = text.replace(" and ", " ").replace(" plus ", "+").split()
    resolved = []
    for p in parts:
        resolved.append(KEY_ALIASES.get(p, p))
    if len(resolved) > 1:
        return "+".join(resolved)
    return text

def parse_natural_to_chain(text):
    """
    Convert natural language instructions into a cc.py chain string.

    Examples:
        "press backspace 5 times, type CLAUDE, then press enter"
        → "key backspace; key backspace; key backspace; key backspace; key backspace; type CLAUDE; key enter"

        "hold control and press a, then press delete"
        → "hold ctrl key a; key delete"

        "click on the save button"
        → "click_text save"

        "scroll down 3 times"
        → "scroll 960 540 -3"

    Returns chain string or None if the text doesn't look like a hot command.
    """
    text = text.strip()
    if not text:
        return None

    # Split on "then", ",", "and then", "after that"
    segments = re.split(r'\s*(?:,\s*then\s*|,\s*and\s+then\s*|,\s*after\s+that\s*|,\s*then\s*|then\s+|,\s*)', text, flags=re.IGNORECASE)
    segments = [s.strip() for s in segments if s.strip()]

    chain_parts = []

    for seg in segments:
        seg_lower = seg.lower()

        # ── "press [key] [N] times" ──
        m = re.match(
            r'(?:press|hit|tap|push)\s+(.+?)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|fifty|hundred|once|twice|thrice)\s*times?',
            seg_lower)
        if m:
            key = _resolve_key(m.group(1))
            count = _parse_number(m.group(2)) or 1
            for _ in range(min(count, 100)):  # Cap at 100 for safety
                chain_parts.append(f"key {key}")
            continue

        # ── "press [key]" (single) ──
        m = re.match(r'(?:press|hit|tap|push)\s+(.+)', seg_lower)
        if m:
            key = _resolve_key(m.group(1))
            chain_parts.append(f"key {key}")
            continue

        # ── "hold [modifier] and press [key]" / "hold [modifier] click [x] [y]" ──
        m = re.match(r'hold\s+(\w+)\s+(?:and\s+)?(?:press|hit|tap)\s+(.+)', seg_lower)
        if m:
            modifier = _resolve_key(m.group(1))
            key = _resolve_key(m.group(2))
            chain_parts.append(f"hold {modifier} key {key}")
            continue

        # ── "type [text]" — use ORIGINAL casing from input ──
        m = re.match(r'(?:type|write|input|enter text)\s+(.+)', seg, flags=re.IGNORECASE)
        if m:
            typed_text = m.group(1).strip()
            # Handle letter-by-letter spelling like "C L A U D E"
            if re.match(r'^[A-Za-z0-9](\s+[A-Za-z0-9])+$', typed_text):
                typed_text = typed_text.replace(" ", "")
            chain_parts.append(f"type {typed_text}")
            continue

        # ── "click on [text]" / "click [text]" ──
        m = re.match(r'(?:click|tap|press)\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+button)?$', seg_lower)
        if m:
            target = m.group(1).strip()
            # If it looks like coordinates (two numbers)
            coord_match = re.match(r'(\d+)\s+(\d+)', target)
            if coord_match:
                chain_parts.append(f"click {coord_match.group(1)} {coord_match.group(2)}")
            else:
                chain_parts.append(f"click_text {target}")
            continue

        # ── "double click on [text]" ──
        m = re.match(r'double\s*click\s+(?:on\s+)?(?:the\s+)?(.+)', seg_lower)
        if m:
            chain_parts.append(f"doubleclick_text {m.group(1).strip()}")
            continue

        # ── "right click on [text]" ──
        m = re.match(r'right\s*click\s+(?:on\s+)?(?:the\s+)?(.+)', seg_lower)
        if m:
            chain_parts.append(f"rightclick_text {m.group(1).strip()}")
            continue

        # ── "scroll up/down [N] times" ──
        m = re.match(r'scroll\s+(up|down)\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)?\s*times?', seg_lower)
        if m:
            direction = -1 if m.group(1) == "down" else 1
            count = _parse_number(m.group(2)) if m.group(2) else 1
            count = count or 1
            chain_parts.append(f"scroll 960 540 {direction * count}")
            continue

        # ── "scroll up/down" (no count) ──
        m = re.match(r'scroll\s+(up|down)', seg_lower)
        if m:
            amount = -3 if m.group(1) == "down" else 3
            chain_parts.append(f"scroll 960 540 {amount}")
            continue

        # ── "wait [N] seconds" ──
        m = re.match(r'wait\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:seconds?)?', seg_lower)
        if m:
            secs = _parse_number(m.group(1)) or 1
            chain_parts.append(f"wait {secs}")
            continue

        # ── "select all" → ctrl+a ──
        if seg_lower.strip() in ["select all", "select everything"]:
            chain_parts.append("key ctrl+a")
            continue

        # ── "copy" / "paste" / "cut" / "undo" / "redo" / "save" ──
        shortcuts = {
            "copy": "ctrl+c", "paste": "ctrl+v", "cut": "ctrl+x",
            "undo": "ctrl+z", "redo": "ctrl+y", "save": "ctrl+s",
            "select all": "ctrl+a", "find": "ctrl+f", "new tab": "ctrl+t",
            "close tab": "ctrl+w", "refresh": "f5",
        }
        if seg_lower.strip() in shortcuts:
            chain_parts.append(f"key {shortcuts[seg_lower.strip()]}")
            continue

        # ── "take a screenshot" / "screenshot" ──
        if "screenshot" in seg_lower:
            chain_parts.append("screenshot")
            continue

        # ── Fallback: if nothing matched, skip this segment ──
        log.warning(f"[PARSE] Could not parse segment: '{seg}'")

    if chain_parts:
        return "; ".join(chain_parts)
    return None

def looks_like_hot_command(text):
    """Heuristic: does this text look like a direct keyboard/mouse instruction?"""
    text_lower = text.lower()
    hot_words = [
        "press ", "hit ", "tap ", "push ",
        "type ", "write ",
        "click ", "double click", "right click",
        "scroll ", "hold ", "select all",
        "copy", "paste", "cut", "undo", "redo", "save",
        "backspace", "enter", "delete", "tab", "escape",
        " times", " then ",
    ]
    return any(w in text_lower for w in hot_words)


# ─── Jarvis Memory (Learn & Unlearn) ────────────────────────────────────────

class JarvisMemory:
    """Persistent memory system — learns from Claude Code responses so repeat tasks are instant.

    Stores learned request→response mappings in a JSON file.
    Uses keyword-based fuzzy matching to find similar past requests.
    Supports unlearning (forget/undo) when a response was wrong.
    """

    MEMORY_FILE = BASE_DIR / "memory.json"
    # Minimum similarity score (0-1) to consider a match
    MATCH_THRESHOLD = 0.75

    def __init__(self):
        self.entries = []
        self._last_entry_id = None  # Track last used entry for unlearning
        self._load()
        log.info(f"[MEM] Loaded {len(self.entries)} learned responses from memory")

    def _load(self):
        if self.MEMORY_FILE.exists():
            try:
                with open(self.MEMORY_FILE, "r") as f:
                    data = json.load(f)
                self.entries = data.get("entries", [])
            except Exception as e:
                log.error(f"[MEM] Failed to load memory: {e}")
                self.entries = []

    def _save(self):
        try:
            with open(self.MEMORY_FILE, "w") as f:
                json.dump({"entries": self.entries}, f, indent=2)
        except Exception as e:
            log.error(f"[MEM] Failed to save memory: {e}")

    def _normalize(self, text: str) -> set:
        """Extract meaningful keywords from text, ignoring filler words."""
        stop_words = {
            "a", "an", "the", "is", "it", "to", "for", "me", "my", "i",
            "can", "you", "please", "just", "and", "or", "of", "in", "on",
            "up", "do", "that", "this", "hey", "jarvis", "could", "would",
            "should", "will", "be", "have", "has", "was", "are", "am",
            "what", "how", "where", "when", "who", "which", "there",
        }
        words = set(re.findall(r'[a-z0-9]+', text.lower()))
        return words - stop_words

    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate keyword overlap similarity between two texts (0-1)."""
        kw1 = self._normalize(text1)
        kw2 = self._normalize(text2)
        if not kw1 or not kw2:
            return 0.0
        intersection = kw1 & kw2
        # Jaccard-like but weighted toward the shorter set (the query)
        smaller = min(len(kw1), len(kw2))
        if smaller == 0:
            return 0.0
        return len(intersection) / smaller

    def lookup(self, user_text: str) -> dict | None:
        """Find a matching learned response for this request. Returns entry or None."""
        best_match = None
        best_score = 0.0

        for entry in self.entries:
            score = self._similarity(user_text, entry["request"])
            if score > best_score and score >= self.MATCH_THRESHOLD:
                best_score = score
                best_match = entry

        if best_match:
            self._last_entry_id = best_match.get("id")
            best_match["use_count"] = best_match.get("use_count", 0) + 1
            best_match["last_used"] = datetime.now().isoformat()
            self._save()
            log.info(f"[MEM] Cache hit! '{user_text[:50]}' matched '{best_match['request'][:50]}' "
                     f"(score: {best_score:.2f}, used {best_match['use_count']}x)")
            return best_match

        return None

    def learn(self, user_text: str, response: str):
        """Store a new request→response mapping in memory."""
        # Don't learn empty, error, or system-error responses
        if not response:
            return
        skip_phrases = [
            "drew a blank", "went wrong", "trouble connecting",
            "cannot find the file", "is not recognized",
            "timed out", "error processing", "apologies, sir",
            "try again", "something simpler",
        ]
        if any(phrase in response.lower() for phrase in skip_phrases):
            log.info(f"[MEM] Skipping learn — response looks like an error")
            return

        # Check if we already have a very similar entry
        for entry in self.entries:
            if self._similarity(user_text, entry["request"]) >= 0.9:
                # Update existing entry instead of creating duplicate
                entry["response"] = response
                entry["updated"] = datetime.now().isoformat()
                entry["use_count"] = entry.get("use_count", 0)
                self._save()
                log.info(f"[MEM] Updated existing memory: '{user_text[:50]}'")
                return

        entry = {
            "id": f"mem_{int(time.time())}_{random.randint(100,999)}",
            "request": user_text,
            "response": response,
            "created": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "use_count": 0,
        }
        self.entries.append(entry)
        self._last_entry_id = entry["id"]
        self._save()
        log.info(f"[MEM] Learned new response for: '{user_text[:50]}' (total: {len(self.entries)} entries)")

    def forget_last(self) -> bool:
        """Remove the last used/learned entry (unlearn). Returns True if something was removed."""
        if not self._last_entry_id:
            return False
        for i, entry in enumerate(self.entries):
            if entry.get("id") == self._last_entry_id:
                removed = self.entries.pop(i)
                self._last_entry_id = None
                self._save()
                log.info(f"[MEM] Forgot: '{removed['request'][:50]}' — removed from memory")
                return True
        return False

    def forget_by_text(self, text: str) -> bool:
        """Remove entries matching this text. Returns True if something was removed."""
        removed_any = False
        self.entries = [
            e for e in self.entries
            if self._similarity(text, e["request"]) < 0.75 or not (removed_any := True)
        ]
        # Simpler approach
        new_entries = []
        for e in self.entries:
            if self._similarity(text, e["request"]) >= 0.75:
                log.info(f"[MEM] Forgot: '{e['request'][:50]}'")
                removed_any = True
            else:
                new_entries.append(e)
        self.entries = new_entries
        if removed_any:
            self._save()
        return removed_any

    def stats(self) -> dict:
        return {
            "total_entries": len(self.entries),
            "most_used": sorted(self.entries, key=lambda e: e.get("use_count", 0), reverse=True)[:5]
        }


# ─── Command Router (Claude is the Brain) ───────────────────────────────────

class CommandRouter:
    """Routes ALL voice commands through Claude as the brain.
    Claude decides what to do — execute cc.py actions, talk, or both."""

    def __init__(self, tts=None):
        self.tts = tts
        self.cc_py = CONFIG.get("cc_py_path", r"C:\Users\tobia\Desktop\ClaudeBridge\skills\cc.py")
        self.custom_commands = self._load_custom_commands()
        self._commands_path = BASE_DIR / CONFIG.get("custom_commands_file", "commands/custom_commands.json")
        self._commands_mtime = self._get_mtime()
        self.memory = JarvisMemory()
        # For voice-based command creation (multi-step state)
        self._creating_command = False
        self._new_cmd_data = {}
        # Prevent multiple LLM responses to same request
        self._processing_request = False
        # Track last request for unlearn
        self._last_user_text = None

    def _get_mtime(self):
        try:
            return self._commands_path.stat().st_mtime
        except Exception:
            return 0

    def _hot_reload_if_changed(self):
        current_mtime = self._get_mtime()
        if current_mtime != self._commands_mtime:
            self._commands_mtime = current_mtime
            self.custom_commands = self._load_custom_commands()
            log.info("[HOT] Commands auto-reloaded (file changed on disk)")

    def _load_custom_commands(self):
        path = BASE_DIR / CONFIG.get("custom_commands_file", "commands/custom_commands.json")
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {"commands": [], "aliases": {}}

    def _run_cc(self, *args):
        """Run cc.py via cmd.exe shell."""
        arg_parts = []
        for a in args:
            if " " in a or ";" in a:
                arg_parts.append(f'"{a}"')
            else:
                arg_parts.append(a)
        cmd_str = f'"{sys.executable}" "{self.cc_py}" {" ".join(arg_parts)}'
        log.info(f"[CMD] Running: cc.py {' '.join(arg_parts)}")
        try:
            result = subprocess.run(cmd_str, shell=True, capture_output=True,
                                    text=True, timeout=CONFIG.get("command_timeout", 30))
            if result.stdout:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {"output": result.stdout}
            if result.stderr:
                log.warning(f"cc.py stderr: {result.stderr}")
            return {"status": "ok"}
        except subprocess.TimeoutExpired:
            log.error("cc.py command timed out")
            return {"status": "error", "message": "timeout"}
        except Exception as e:
            log.error(f"cc.py error: {e}")
            return {"status": "error", "message": str(e)}
    
    # Path to scene_parser.py for rich screen understanding
    SCENE_PARSER = Path(CONFIG.get("cc_py_path", "")).parent / "scene_parser.py"

    def _visual_find_and_click(self, description: str, click_type: str = "click") -> bool:
        """Use scene_parser + Claude Code CLI (multimodal) to find a visual element and click it.

        Pipeline:
        1. Run scene_parser.py --fast to get screenshot + element map (OCR, contours, symbols)
        2. Send the annotated screenshot + element map to Claude Code CLI
        3. Claude identifies the target element and returns coordinates
        4. Click at those coordinates

        Args:
            description: What to find, e.g. "the Defect character", "the green checkmark"
            click_type: "click", "doubleclick", or "rightclick"
        Returns:
            True if element was found and clicked, False otherwise
        """
        try:
            # Step 1: Run scene_parser to capture + analyze the screen
            log.info(f"[VISUAL] Running scene_parser to find: '{description}'")
            t_start = time.time()

            scene_parser_path = self.SCENE_PARSER
            if not scene_parser_path.exists():
                # Fallback paths
                for p in [
                    Path(r"C:\Users\tobia\Desktop\ClaudeBridge\skills\scene_parser.py"),
                    Path(CONFIG.get("cc_py_path", "")).parent / "scene_parser.py",
                ]:
                    if p.exists():
                        scene_parser_path = p
                        break

            if not scene_parser_path.exists():
                log.error(f"[VISUAL] scene_parser.py not found")
                # Fall back to screenshot-only approach
                return self._visual_find_screenshot_only(description, click_type)

            # Run scene_parser --fast --no-email for quick OCR + elements
            sp_result = subprocess.run(
                f'python "{scene_parser_path}" --fast --no-email',
                shell=True, capture_output=True, text=True, timeout=30,
            )

            sp_elapsed = time.time() - t_start
            log.info(f"[VISUAL] scene_parser completed in {sp_elapsed:.1f}s")

            # Parse scene_parser JSON output
            scene_data = None
            if sp_result.stdout:
                try:
                    scene_data = json.loads(sp_result.stdout.strip())
                except json.JSONDecodeError:
                    log.warning("[VISUAL] Could not parse scene_parser JSON output")

            # Get paths from scene data or find latest files
            screenshot_dir = Path(r"C:\Users\tobia\Desktop\ClaudeBridge\screenshots")
            scan_image = None
            element_map_path = screenshot_dir / "last_element_map.json"

            if scene_data:
                scan_image = scene_data.get("scan_image") or scene_data.get("raw_image")
                elements = scene_data.get("elements", [])
                scene_summary = scene_data.get("scene_summary", "")
            else:
                # Load from saved element map file
                if element_map_path.exists():
                    try:
                        map_data = json.loads(element_map_path.read_text())
                        scan_image = map_data.get("scan_image") or map_data.get("raw_image")
                        elements = map_data.get("elements", [])
                        scene_summary = map_data.get("scene_summary", "")
                    except:
                        elements = []
                        scene_summary = ""
                else:
                    elements = []
                    scene_summary = ""

            if not scan_image:
                # Find latest screenshot as fallback
                shots = sorted(screenshot_dir.glob("scan_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                if shots:
                    scan_image = str(shots[0])
                else:
                    shots = sorted(screenshot_dir.glob("screenshot_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    scan_image = str(shots[0]) if shots else None

            if not scan_image:
                log.error("[VISUAL] No screenshot available")
                return False

            log.info(f"[VISUAL] Got {len(elements)} elements from scene_parser")

            # Step 2: Build a compact element summary for Claude
            # Keep it short: only HIGH confidence + first 40 elements to stay under limits
            element_summary = ""
            if elements:
                lines = []
                # Prioritize HIGH confidence elements, then MED
                sorted_els = sorted(elements, key=lambda e: (
                    0 if e.get("confidence_tier") == "HIGH" else
                    1 if e.get("confidence_tier") == "MED" else 2
                ))
                for el in sorted_els[:40]:
                    label = el.get("label", "")
                    el_class = el.get("element_class", "")
                    cx, cy = el.get("cx", 0), el.get("cy", 0)
                    el_type = el.get("type", "")
                    lines.append(f"  {el_type}/{el_class}: '{label}' @ ({cx},{cy})")
                element_summary = "\n".join(lines)

            # Step 3: Ask Claude Code CLI to identify the target
            # Write prompt to a temp file to avoid Windows command-line length limit (8191 chars)
            log.info(f"[VISUAL] Asking Claude Code to locate '{description}' using screenshot + element map")
            t_claude = time.time()

            vision_prompt = (
                f'Look at this screenshot: {scan_image}\n\n'
                f'Here is the element map from scene analysis:\n'
                f'{scene_summary}\n\n'
                f'Detected elements:\n{element_summary}\n\n'
                f'Find the CENTER pixel coordinates of: {description}\n'
                f'Reply with ONLY two numbers: X Y (e.g. "450 320").\n'
                f'If you find a matching element in the list above, use its cx/cy coordinates.\n'
                f'Otherwise, look at the screenshot image itself to find it visually.\n'
                f'If you cannot find it at all, reply with just: NOT_FOUND\n'
                f'No other text or explanation — just the coordinates or NOT_FOUND.'
            )

            # Use Popen with stdin pipe to avoid Windows 8191-char command line limit
            result = subprocess.run(
                [str(self.CLAUDE_CMD), "-p", "-", "--output-format", "text"],
                input=vision_prompt,
                capture_output=True, text=True, timeout=45,
                cwd=str(screenshot_dir),
            )

            c_elapsed = time.time() - t_claude
            coords_text = (result.stdout.strip() if result.stdout else "").strip()
            if not coords_text and result.stderr:
                coords_text = result.stderr.strip()

            total_elapsed = time.time() - t_start
            log.info(f"[VISUAL] Claude Code response ({c_elapsed:.1f}s, total: {total_elapsed:.1f}s): '{coords_text}'")

            if not coords_text or "not_found" in coords_text.lower():
                log.warning(f"[VISUAL] Element not found on screen: '{description}'")
                return False

            # Parse coordinates — find the first pair of numbers in the response
            coord_match = re.search(r'(\d{2,4})\s+(\d{2,4})', coords_text)
            if not coord_match:
                coord_match = re.search(r'(\d{2,4})\s*,\s*(\d{2,4})', coords_text)
            if not coord_match:
                log.warning(f"[VISUAL] Could not parse coordinates from: '{coords_text}'")
                return False

            x, y = int(coord_match.group(1)), int(coord_match.group(2))
            log.info(f"[VISUAL] Found '{description}' at ({x}, {y}) — clicking...")

            # Step 4: Click at the coordinates
            if click_type == "doubleclick":
                self._run_cc("chain", f"click {x} {y}; wait 0.1; click {x} {y}")
            elif click_type == "rightclick":
                self._run_cc("chain", f"rightclick {x} {y}")
            else:
                self._run_cc("chain", f"click {x} {y}")

            log.info(f"[VISUAL] Successfully clicked '{description}' at ({x}, {y})")
            return True

        except subprocess.TimeoutExpired:
            log.warning("[VISUAL] Timed out during visual find")
            return False
        except Exception as e:
            log.error(f"[VISUAL] Error during visual find & click: {e}")
            return False

    def _visual_find_screenshot_only(self, description: str, click_type: str = "click") -> bool:
        """Fallback: Use just a screenshot + Claude Code CLI when scene_parser is unavailable."""
        try:
            log.info(f"[VISUAL] Fallback: screenshot-only mode for '{description}'")
            self._run_cc("chain", "screenshot")

            screenshot_dir = Path(r"C:\Users\tobia\Desktop\ClaudeBridge\screenshots")
            screenshots = sorted(screenshot_dir.glob("screenshot_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not screenshots:
                log.error("[VISUAL] No screenshots found")
                return False

            screenshot_path = screenshots[0]
            vision_prompt = (
                f'Look at this screenshot: {screenshot_path} '
                f'Find the CENTER pixel coordinates of: {description} '
                f'Reply with ONLY two numbers: X Y (e.g. "450 320"). '
                f'If you cannot find it, reply: NOT_FOUND. No other text.'
            )
            safe_prompt = vision_prompt.replace('"', '\\"')

            result = subprocess.run(
                f'"{self.CLAUDE_CMD}" -p "{safe_prompt}" --output-format text',
                shell=True, capture_output=True, text=True, timeout=30,
                cwd=str(screenshot_dir),
            )

            coords_text = (result.stdout.strip() if result.stdout else "").strip()
            if not coords_text or "not_found" in coords_text.lower():
                return False

            coord_match = re.search(r'(\d{2,4})\s+(\d{2,4})', coords_text)
            if not coord_match:
                coord_match = re.search(r'(\d{2,4})\s*,\s*(\d{2,4})', coords_text)
            if not coord_match:
                return False

            x, y = int(coord_match.group(1)), int(coord_match.group(2))
            log.info(f"[VISUAL] Fallback found '{description}' at ({x}, {y})")

            if click_type == "doubleclick":
                self._run_cc("chain", f"click {x} {y}; wait 0.1; click {x} {y}")
            elif click_type == "rightclick":
                self._run_cc("chain", f"rightclick {x} {y}")
            else:
                self._run_cc("chain", f"click {x} {y}")
            return True

        except Exception as e:
            log.error(f"[VISUAL] Fallback error: {e}")
            return False

    def git_push(self):
        """Git push with error handling."""
        log.info("[GIT] Pushing changes...")
        try:
            # First check if we're in a git repo
            result = subprocess.run(
                "git status",
                shell=True, capture_output=True, text=True,
                timeout=30, cwd=BASE_DIR
            )
            
            if "not a git repository" in result.stderr:
                log.warning("[GIT] Not in a git repository")
                return {"status": "error", "message": "Not a git repository"}
            
            # Push changes
            result = subprocess.run(
                "git push",
                shell=True, capture_output=True, text=True,
                timeout=60, cwd=BASE_DIR
            )
            
            if result.returncode == 0:
                log.info("[GIT] Push successful")
                return {"status": "ok", "message": "Changes pushed successfully"}
            else:
                log.warning(f"[GIT] Push failed: {result.stderr}")
                return {"status": "error", "message": result.stderr}
                
        except Exception as e:
            log.error(f"[GIT] Push error: {e}")
            return {"status": "error", "message": str(e)}

    def _execute_action(self, action_str: str):
        """Execute a single [ACTION: ...] line from Claude's response."""
        action_str = action_str.strip()

        if action_str == "speak_only":
            log.info("[ACT] Speak only — no PC action needed")
            return

        # chain "command1; command2; ..."
        m = re.match(r'chain\s+"(.+)"', action_str)
        if not m:
            m = re.match(r"chain\s+'(.+)'", action_str)
        if m:
            chain_cmd = m.group(1)
            log.info(f"[ACT] Executing chain: {chain_cmd}")
            # Split chain into segments and handle visual_click commands locally
            segments = [s.strip() for s in chain_cmd.split(";")]
            non_visual_batch = []
            for seg in segments:
                vc_match = re.match(r'visual_click\s+(.+)', seg, re.IGNORECASE)
                dvc_match = re.match(r'visual_doubleclick\s+(.+)', seg, re.IGNORECASE)
                rvc_match = re.match(r'visual_rightclick\s+(.+)', seg, re.IGNORECASE)
                if vc_match or dvc_match or rvc_match:
                    # Flush any queued non-visual commands first
                    if non_visual_batch:
                        self._run_cc("chain", "; ".join(non_visual_batch))
                        non_visual_batch = []
                    # Execute the visual click
                    if vc_match:
                        self._visual_find_and_click(vc_match.group(1).strip())
                    elif dvc_match:
                        self._visual_find_and_click(dvc_match.group(1).strip(), click_type="doubleclick")
                    elif rvc_match:
                        self._visual_find_and_click(rvc_match.group(1).strip(), click_type="rightclick")
                else:
                    non_visual_batch.append(seg)
            # Flush remaining non-visual commands
            if non_visual_batch:
                self._run_cc("chain", "; ".join(non_visual_batch))
            return

        # claude_code "prompt"
        m = re.match(r'claude_code\s+"(.+)"', action_str)
        if not m:
            m = re.match(r"claude_code\s+'(.+)'", action_str)
        if m:
            prompt = m.group(1)
            log.info(f"[ACT] Delegating to Claude Code: {prompt[:80]}...")
            try:
                subprocess.Popen(
                    f'claude -p "{prompt.replace(chr(34), chr(39))}"',
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=CONFIG.get("claude_code_workdir", os.path.expanduser("~\\Desktop")),
                )
            except Exception as e:
                log.error(f"[ACT] Claude Code delegation failed: {e}")
            return

        log.warning(f"[ACT] Unknown action format: {action_str}")

    def _parse_and_execute(self, response: str):
        """Parse Claude's response for [ACTION] and [SPEAK] tags, execute actions, return speech."""
        actions = re.findall(r'\[ACTION:\s*(.+?)\]', response)
        speak_match = re.search(r'\[SPEAK\]\s*(.+?)(?:\[|$)', response, re.DOTALL)

        if not speak_match:
            # Fallback: if Claude didn't use the format, treat entire response as speech
            # Strip any accidental tags
            clean = re.sub(r'\[ACTION:.*?\]', '', response).strip()
            clean = re.sub(r'\[SPEAK\]', '', clean).strip()
            speak_text = clean if clean else response.strip()
        else:
            speak_text = speak_match.group(1).strip()

        # Execute all actions
        for action in actions:
            try:
                self._execute_action(action)
            except Exception as e:
                log.error(f"[ACT] Action failed: {e}")

        return speak_text

    def route(self, text: str) -> dict:
        """Route everything through Claude as the brain."""
        if not text:
            return {"status": "empty", "message": "No speech detected"}

        self._hot_reload_if_changed()
        text_lower = text.lower().strip()
        log.info(f"[>>] Processing: \"{text}\"")

        # ── Voice command creation flow ──
        if self._creating_command:
            return self._voice_create_step(text)

        # ── Local-only commands (no need to bother Claude) ──
        if text_lower in ["stop", "quit", "exit", "goodbye", "shut down"]:
            if self.tts:
                self.tts.speak("Shutting down. Try not to miss me too much.")
            return {"status": "exit", "message": "Shutting down Jarvis"}

        if any(text_lower.startswith(p) for p in ["reload commands", "refresh commands"]):
            self.custom_commands = self._load_custom_commands()
            self._commands_mtime = self._get_mtime()
            if self.tts:
                self.tts.speak_async("Commands reloaded, sir.")
            return {"status": "ok", "action": "reload_commands"}

        if any(text_lower.startswith(p) for p in [
            "open command manager", "command manager", "open commands",
            "manage commands", "open manager", "open enhanced manager"]):
            gui_path = BASE_DIR / "jarvis_gui_enhanced.py"
            subprocess.Popen([sys.executable, str(gui_path)],
                            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
            if self.tts:
                self.tts.speak_async("Opening the enhanced command manager.")
            return {"status": "ok", "action": "open_enhanced_command_manager"}

        if any(text_lower.startswith(p) for p in [
            "add command", "new command", "create command",
            "add a command", "create a command"]):
            return self._start_voice_create(text_lower)

        # ── Unlearn / Forget commands ──
        if any(text_lower.startswith(p) for p in [
            "that was wrong", "that's wrong", "forget that", "undo that",
            "unlearn that", "wrong answer", "that was bad", "no that's wrong",
            "nope", "that wasn't right", "not what i asked"]):
            if self.memory.forget_last():
                if self.tts:
                    self.tts.speak_async("Noted. I've forgotten that response. I'll ask Claude Code fresh next time.")
                return {"status": "ok", "action": "unlearn"}
            else:
                if self.tts:
                    self.tts.speak_async("Nothing to forget at the moment, sir.")
                return {"status": "ok", "action": "unlearn_empty"}

        if text_lower.startswith("forget how to ") or text_lower.startswith("unlearn "):
            forget_text = text_lower.replace("forget how to ", "").replace("unlearn ", "").strip()
            if self.memory.forget_by_text(forget_text):
                if self.tts:
                    self.tts.speak_async(f"Forgotten everything about {forget_text}.")
            else:
                if self.tts:
                    self.tts.speak_async(f"I don't have anything stored about that.")
            return {"status": "ok", "action": "unlearn_specific"}

        if text_lower in ["what have you learned", "show memory", "memory stats",
                          "what do you know", "how many things have you learned"]:
            stats = self.memory.stats()
            count = stats['total_entries']
            if self.tts:
                self.tts.speak_async(f"I've learned {count} response{'s' if count != 1 else ''} so far, sir.")
            return {"status": "ok", "action": "memory_stats", **stats}

        if any(text_lower.startswith(p) for p in [
            "recent memory", "show recent memory", "what do you remember",
            "list memory", "read memory", "read back memory"]):
            entries = self.memory.entries
            if not entries:
                if self.tts:
                    self.tts.speak_async("My memory is empty at the moment. I haven't learned anything yet.")
                return {"status": "ok", "action": "memory_empty"}
            # Show last 5 entries
            recent = entries[-5:]
            lines = []
            for e in recent:
                req = e["request"][:60]
                uses = e.get("use_count", 0)
                lines.append(f"{req} — used {uses} time{'s' if uses != 1 else ''}")
            summary = ". ".join(lines)
            # Also log the full details to console
            log.info("[MEM] Recent memory entries:")
            for e in recent:
                log.info(f"  - Request: {e['request'][:80]}")
                log.info(f"    Uses: {e.get('use_count', 0)}, Created: {e.get('created', '?')}")
            if self.tts:
                count = len(entries)
                self.tts.speak_async(
                    f"I have {count} thing{'s' if count != 1 else ''} in memory. "
                    f"Here are the most recent: {summary}"
                )
            return {"status": "ok", "action": "memory_list", "entries": recent}

        if text_lower in ["clear memory", "wipe memory", "forget everything",
                          "clear all memory", "reset memory"]:
            count = len(self.memory.entries)
            self.memory.entries = []
            self.memory._save()
            if self.tts:
                self.tts.speak_async(f"Memory wiped. {count} entries cleared. Starting fresh.")
            return {"status": "ok", "action": "memory_cleared", "cleared": count}

        # ── Check custom commands (fast local match before hitting Claude) ──
        for cmd in self.custom_commands.get("commands", []):
            for trigger in cmd.get("triggers", []):
                if text_lower.startswith(trigger.lower()):
                    action = cmd.get("action", "")
                    if action.startswith("chain:"):
                        chain_cmd = action[6:]
                        if self.tts:
                            self.tts.speak_async("On it.")
                        return self._run_cc("chain", chain_cmd)

        # ── Check Memory (instant cached responses) ──
        self._last_user_text = text
        cached = self.memory.lookup(text)
        if cached:
            log.info(f"[MEM] Using cached response (no Claude Code call needed)")
            # Handle the cached response just like a fresh one
            def _handle_cached():
                self._handle_response(cached["response"])
            thread = threading.Thread(target=_handle_cached, daemon=True)
            thread.start()
            return {"status": "ok", "action": "memory_hit", "prompt": text}

        # ── Everything else → Claude Brain ──
        log.info("[BRAIN] Sending to Claude for interpretation...")
        self._ask_claude(text)
        return {"status": "ok", "action": "claude_brain", "prompt": text}

    def _ask_claude(self, user_text: str):
        """Send user speech to Claude SDK with Claude API and GPT-4o-mini fallbacks. Runs in background thread."""
        def _run():
            try:
                # Prevent multiple responses to the same request
                if hasattr(self, '_processing_request') and self._processing_request:
                    log.warning("[BRAIN] Already processing request, ignoring duplicate")
                    return
                
                self._processing_request = True
                
                now = datetime.now()
                time_hint = now.strftime("%I:%M %p on %A, %B %d, %Y")
                system_prompt = JARVIS_SYSTEM_PROMPT.replace("{time_hint}", time_hint)
                system_prompt += f"\n\nCurrent time: {time_hint}\n\nUser said: {user_text}"

                # Claude Code CLI — PRIMARY (can actually do things on this machine)
                log.info(f"[BRAIN] Using Claude Code CLI (primary)...")
                response_text = self._try_claude_code_cli(user_text, system_prompt)
                if response_text:
                    self._handle_response(response_text)
                    return
                else:
                    log.warning("[BRAIN] Claude Code CLI failed, trying GPT-4o-mini...")

                # GPT-4o-mini — fast fallback for when Claude Code is unavailable
                if HAS_OPENAI_FALLBACK and openai_fallback_client:
                    log.info(f"[BRAIN] Using GPT-4o-mini (fallback)...")
                    response_text = self._try_openai_fallback(user_text, system_prompt)
                    if response_text:
                        self._handle_response(response_text)
                        return
                    else:
                        log.warning("[BRAIN] GPT-4o-mini also failed")
                
                # Nothing worked
                log.error("[BRAIN] All AI methods failed")
                if self.tts:
                    self.tts.speak("I'm having trouble connecting right now, sir.")

            except Exception as e:
                log.error(f"[BRAIN] Error: {e}")
                if self.tts:
                    self.tts.speak("Something went wrong on my end. Apologies, sir.")
            finally:
                self._processing_request = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
    
    # Full path to claude CLI (npm global install)
    CLAUDE_CMD = CONFIG.get("claude_path", r"C:\Users\tobia\AppData\Roaming\npm\claude.cmd")

    def _try_claude_code_cli(self, user_text: str, system_prompt: str) -> str:
        """Use the local Claude Code CLI (claude -p) — can actually run commands on this machine."""
        try:
            input_tokens = count_tokens(f"{system_prompt}\n\n{user_text}", "claude-3-sonnet")

            # Build prompt: system instructions + user request
            full_prompt = f"{system_prompt}\n\nUser said: {user_text}"
            # Escape for shell safety
            safe_prompt = full_prompt.replace('"', '\\"').replace('\n', ' ').replace('\r', '')

            t_start = time.time()
            result = subprocess.run(
                f'"{self.CLAUDE_CMD}" -p "{safe_prompt}" --output-format text',
                shell=True, capture_output=True, text=True, timeout=60,
                cwd=CONFIG.get("claude_code_workdir", os.path.expanduser("~\\Desktop")),
            )
            elapsed = time.time() - t_start

            response_text = result.stdout.strip() if result.stdout else ""
            if not response_text and result.stderr:
                response_text = result.stderr.strip()

            # Filter out Windows/system errors that aren't real responses
            error_phrases = [
                "cannot find the file specified",
                "is not recognized as an internal",
                "not recognized as a cmdlet",
                "access is denied",
            ]
            if response_text and any(err in response_text.lower() for err in error_phrases):
                log.warning(f"[BRAIN] Claude Code CLI returned system error: {response_text[:100]}")
                return ""

            log.info(f"[BRAIN] Claude Code CLI responded in {elapsed:.1f}s: '{response_text[:100] if response_text else 'EMPTY'}'")

            if response_text:
                output_tokens = count_tokens(response_text, "claude-3-sonnet")
                log_tokens("claude", input_tokens, output_tokens)

            return response_text

        except subprocess.TimeoutExpired:
            log.warning("[BRAIN] Claude Code CLI timed out after 60s")
            return ""
        except Exception as e:
            log.debug(f"[BRAIN] Claude Code CLI error: {e}")
            return ""

    def _try_claude_sdk(self, user_text: str, system_prompt: str) -> str:
        """Try Claude Code Agent SDK (primary method)."""
        try:
            # Count input tokens
            input_text = f"{system_prompt}\n\n{user_text}"
            input_tokens = count_tokens(input_text, "claude-3-sonnet")
            
            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                permission_mode="auto",
                model="sonnet",
                output_format="text",
                temperature=0.85,
            )
            
            import asyncio
            
            async def _claude_query():
                response_text = ""
                async for message in query(prompt=user_text, options=options):
                    if hasattr(message, 'type'):
                        if message.type == 'content_block' and hasattr(message, 'content'):
                            if hasattr(message.content, 'text'):
                                response_text += message.content.text
                            elif isinstance(message.content, str):
                                response_text += message.content
                        elif message.type == 'text' and hasattr(message, 'text'):
                            response_text += message.text
                    else:
                        if hasattr(message, 'content'):
                            if hasattr(message.content, 'text'):
                                response_text += message.content.text
                            elif isinstance(message.content, str):
                                response_text += message.content
                        elif hasattr(message, 'text'):
                            response_text += message.text
                
                return response_text.strip()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_text = loop.run_until_complete(_claude_query())
            loop.close()
            
            # Count output tokens and log usage
            output_tokens = count_tokens(response_text, "claude-3-sonnet")
            log_tokens("claude", input_tokens, output_tokens)
            
            return response_text
            
        except Exception as e:
            log.debug(f"[BRAIN] Claude SDK error: {e}")
            return ""
    
    def _try_claude_api(self, user_text: str, system_prompt: str) -> str:
        """Try Claude API (secondary fallback)."""
        try:
            # Count input tokens
            input_text = f"{system_prompt}\n\n{user_text}"
            input_tokens = count_tokens(input_text, "claude-3-sonnet")
            
            response = claude_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                temperature=0.85,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_text}
                ]
            )
            
            response_text = response.content[0].text.strip()
            
            # Count output tokens and log usage
            output_tokens = count_tokens(response_text, "claude-3-sonnet")
            log_tokens("claude_api", input_tokens, output_tokens)
            
            return response_text
            
        except Exception as e:
            log.debug(f"[BRAIN] Claude API error: {e}")
            return ""
    
    def _try_openai_fallback(self, user_text: str, system_prompt: str) -> str:
        """Try OpenAI GPT-4o-mini (cheap fallback)."""
        try:
            # Count input tokens
            input_text = f"{system_prompt}\n\n{user_text}"
            input_tokens = count_tokens(input_text, "gpt-4o-mini")
            
            response = openai_fallback_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=500,
                temperature=0.85
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Count output tokens and log usage
            output_tokens = count_tokens(response_text, "gpt-4o-mini")
            log_tokens("openai_chat", input_tokens, output_tokens)
            
            return response_text
            
        except Exception as e:
            log.debug(f"[BRAIN] GPT-4o-mini error: {e}")
            return ""
    
    def _handle_response(self, response_text: str):
        """Process AI response — run TTS and actions in PARALLEL for faster feel."""
        calling_thread = threading.current_thread().name
        log.info(f"[BRAIN] _handle_response called (thread: {calling_thread})")

        if not response_text:
            log.warning("[BRAIN] Empty response")
            if self.tts:
                self.tts.speak("I seem to have drawn a blank on that one. Could you try again?")
            return

        log.info(f"[BRAIN] Response: {response_text[:300]}...")

        # Parse out actions and speech text
        actions = re.findall(r'\[ACTION:\s*(.+?)\]', response_text)
        speak_match = re.search(r'\[SPEAK\]\s*(.+?)(?:\[|$)', response_text, re.DOTALL)

        if not speak_match:
            clean = re.sub(r'\[ACTION:.*?\]', '', response_text).strip()
            clean = re.sub(r'\[SPEAK\]', '', clean).strip()
            speak_text = clean if clean else response_text.strip()
        else:
            speak_text = speak_match.group(1).strip()

        # Run TTS and actions BOTH in separate threads — fully parallel
        tts_thread = None
        if speak_text and self.tts:
            log.info(f"[BRAIN] Speaking + executing in parallel...")
            tts_thread = self.tts.speak_async(speak_text)

        # Execute actions in their own thread so nothing blocks
        def _run_actions():
            for action in actions:
                try:
                    self._execute_action(action)
                except Exception as e:
                    log.error(f"[ACT] Action failed: {e}")

        if actions and not any("speak_only" in a for a in actions):
            action_thread = threading.Thread(target=_run_actions, daemon=True)
            action_thread.start()

        # Wait for TTS to finish so wake word detection doesn't restart too early
        if tts_thread and tts_thread.is_alive():
            tts_thread.join(timeout=30)

        # Auto-learn: save this response to memory for instant replay next time
        if self._last_user_text and response_text:
            self.memory.learn(self._last_user_text, response_text)

    # ─── Voice Command Creation ──────────────────────────────────────────

    def _start_voice_create(self, text: str):
        log.info("[+] Starting enhanced voice command creation...")
        self._creating_command = True
        self._new_cmd_data = {"step": "name", "gui_mode": True}
        if self.tts:
            self.tts.speak_async("Alright, let's create a new command. You can use voice or open the enhanced GUI for visual editing.")
        return {"status": "listening", "action": "voice_create", "message": "Say the command name or say 'open GUI' for visual editing"}

    def _voice_create_step(self, text: str):
        step = self._new_cmd_data.get("step", "name")

        if text.lower().strip() in ["cancel", "stop", "nevermind", "never mind"]:
            self._creating_command = False
            self._new_cmd_data = {}
            if self.tts:
                self.tts.speak_async("Command creation cancelled.")
            return {"status": "cancelled", "action": "voice_create"}
        
        # Check for GUI mode request
        if "gui" in text.lower() or "visual" in text.lower():
            gui_path = BASE_DIR / "jarvis_gui_enhanced.py"
            subprocess.Popen([sys.executable, str(gui_path)],
                            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
            if self.tts:
                self.tts.speak_async("Opening the enhanced command manager for visual editing.")
            self._creating_command = False
            self._new_cmd_data = {}
            return {"status": "opened_gui", "action": "voice_create"}

        if step == "name":
            self._new_cmd_data["name"] = text.strip()
            self._new_cmd_data["step"] = "trigger"
            if self.tts:
                self.tts.speak_async(f"Got it, {text.strip()}. Now say the trigger phrase.")
            return {"status": "listening", "step": "trigger"}

        elif step == "trigger":
            self._new_cmd_data["trigger"] = text.strip().lower()
            self._new_cmd_data["step"] = "action"
            if self.tts:
                self.tts.speak_async(f"Trigger set. Now tell me the action.")
            return {"status": "listening", "step": "action"}

        elif step == "action":
            action_text = text.strip().lower()
            action_map = {
                "prompt cowork": "prompt_cowork", "prompt claude code": "prompt_claude_code",
                "open cowork": "open_cowork", "open terminal": "open_terminal",
                "screenshot": "screenshot", "type text": "type_text",
            }
            action = action_map.get(action_text, action_text)
            if action_text.startswith("chain"):
                action = f"chain:{action_text[5:].strip()}"

            new_cmd = {
                "name": self._new_cmd_data["name"],
                "triggers": [self._new_cmd_data["trigger"]],
                "action": action,
                "description": "Voice-created command"
            }
            self.custom_commands.setdefault("commands", []).append(new_cmd)
            commands_path = BASE_DIR / CONFIG.get("custom_commands_file", "commands/custom_commands.json")
            with open(commands_path, "w") as f:
                json.dump(self.custom_commands, f, indent=4)

            self._creating_command = False
            self._new_cmd_data = {}
            if self.tts:
                self.tts.speak_async(f"Command created. {new_cmd['name']} is live.")
            return {"status": "ok", "action": "voice_create_complete", "command": new_cmd}

        self._creating_command = False
        return {"status": "error", "message": "Unknown step"}


# ─── Visual Feedback ──────────────────────────────────────────────────────────

def show_visual_notification(message: str, duration_ms: 100):
    """Show a very brief visual flash when Jarvis is listening."""
    try:
        import tkinter as tk
        
        # Create a simple window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Create notification window - very small and subtle
        notification = tk.Toplevel(root)
        notification.title("")
        notification.geometry("60x20+600+300")  # Small size
        notification.configure(bg='#2a2a2a')  # Dark gray
        
        # Remove window decorations for clean look
        notification.overrideredirect(True)
        
        # Add a simple colored indicator
        label = tk.Label(
            notification,
            text="●",  # Just a dot
            fg='#00ff41',  # Jarvis green
            bg='#2a2a2a',
            font=('Consolas', 12)
        )
        label.pack(expand=True, fill='both')
        
        # Center on screen
        notification.update_idletasks()
        width = notification.winfo_width()
        height = notification.winfo_height()
        x = (notification.winfo_screenwidth() // 2) - (width // 2)
        y = (notification.winfo_screenheight() // 2) - (height // 2)
        notification.geometry(f'{width}x{height}+{x}+{y}')
        
        # Auto-close after very short duration
        notification.after(duration_ms, notification.destroy)
        
        # Make it always on top
        notification.attributes('-topmost', True)
        
        # Start the GUI event loop in a separate thread
        import threading
        
        def run_gui():
            try:
                root.mainloop()
            except:
                pass  # Ignore any GUI errors
        
        gui_thread = threading.Thread(target=run_gui, daemon=True)
        gui_thread.start()
        
    except Exception as e:
        log.debug(f"[VISUAL] Could not show notification: {e}")
        # Silent fallback - no console output for subtlety


# ─── Text-to-Speech Engine ──────────────────────────────────────────────────

class JarvisTTS:
    """Neural TTS via Microsoft Edge (free). Smooth British male voice."""

    VOICE = "en-GB-RyanNeural"

    def __init__(self):
        self._lock = threading.Lock()
        self._tts_dir = BASE_DIR / "tts_cache"
        self._tts_dir.mkdir(exist_ok=True)
        self.available = HAS_EDGE_TTS
        self._is_speaking = False  # Prevent duplicate playback
        
        # Initialize pygame mixer for faster audio playback
        if HAS_PYGAME:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                log.info("[TTS] Pygame mixer initialized for fast audio playback")
            except Exception as e:
                log.warning(f"[TTS] Pygame init failed: {e}")
        
        if self.available:
            log.info(f"[TTS] Edge TTS initialized (voice: {self.VOICE}) — neural, free")
        else:
            log.warning("[TTS] edge-tts not installed — TTS disabled. Run: pip install edge-tts")

    def _clean_text(self, text: str) -> str:
        clean = text.strip()
        clean = clean.replace("```", "").replace("`", "")
        clean = clean.replace("**", "").replace("__", "")
        clean = clean.replace("#", "")
        # Strip any leftover action tags
        clean = re.sub(r'\[ACTION:.*?\]', '', clean)
        clean = re.sub(r'\[SPEAK\]', '', clean)
        clean = re.sub(r'https?://\S+', '', clean)
        clean = clean.strip()
        return clean

    def speak(self, text: str):
        if not self.available or not text:
            return
        clean = self._clean_text(text)
        if not clean:
            return

        # Enhanced logging to track duplicate calls
        calling_thread = threading.current_thread().name
        log.info(f"[TTS] Speaking: {clean[:100]}... (thread: {calling_thread})")
        
        # Thread-safe duplicate prevention
        with self._lock:
            if hasattr(self, '_is_speaking') and self._is_speaking:
                log.warning(f"[TTS] Already speaking, skipping duplicate (thread: {calling_thread})")
                return
            self._is_speaking = True
        
        try:
            # Use timestamp to avoid file conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            audio_path = self._tts_dir / f"jarvis_response_{timestamp}.mp3"
            
            log.info(f"[TTS] Generating audio file: {audio_path}")

            async def _generate():
                communicate = edge_tts.Communicate(clean, self.VOICE, rate="-5%")
                await communicate.save(str(audio_path))

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_generate())
            loop.close()
            
            log.info(f"[TTS] Audio file generated: {audio_path} (size: {audio_path.stat().st_size} bytes)")

            # Fast pygame playback instead of PowerShell
            if HAS_PYGAME and audio_path.exists():
                try:
                    log.info(f"[TTS] Using pygame playback for: {audio_path}")
                    
                    # Stop any currently playing music first
                    if pygame.mixer.music.get_busy():
                        log.warning("[TTS] Stopping currently playing music")
                        pygame.mixer.music.stop()
                        pygame.time.wait(200)  # Brief pause to ensure stop
                    
                    pygame.mixer.music.load(str(audio_path))
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.wait(100)
                    
                    log.info(f"[TTS] Pygame playback finished for: {audio_path}")
                        
                except Exception as e:
                    log.warning(f"[TTS] Pygame playback failed, falling back: {e}")
                    # Fallback to original method
                    self._fallback_playback(str(audio_path))
            else:
                # Original fallback method
                log.info(f"[TTS] Using fallback playback for: {audio_path}")
                self._fallback_playback(str(audio_path))
            
            # Clean up old audio files (keep only last 5)
            try:
                self._cleanup_old_audio()
            except Exception as e:
                log.debug(f"[TTS] Cleanup warning: {e}")
                

        except Exception as e:
            log.error(f"[TTS] Speech error: {e}")
        finally:
            with self._lock:
                self._is_speaking = False
                log.debug(f"[TTS] Finished speaking (thread: {calling_thread})")
    
    def _fallback_playback(self, audio_path: str):
        """Original PowerShell/ffplay fallback method."""
        if os.name == 'nt':
            ps_cmd = (
                f"$p = New-Object System.Windows.Media.MediaPlayer; "
                f"$p.Open([Uri]'{audio_path}'); "
                f"$p.Play(); "
                f"Start-Sleep -Milliseconds 500; "
                f"while ($p.NaturalDuration.HasTimeSpan -eq $false) {{ Start-Sleep -Milliseconds 100 }}; "
                f"while ($p.Position -lt $p.NaturalDuration.TimeSpan) {{ Start-Sleep -Milliseconds 100 }}; "
                f"$p.Close()"
            )
            subprocess.run(
                ["powershell", "-Command",
                 f"Add-Type -AssemblyName PresentationCore; {ps_cmd}"],
                shell=False, timeout=60,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", audio_path],
                           timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def speak_async(self, text: str):
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t


# ─── Main Loop ───────────────────────────────────────────────────────────────

class Jarvis:
    """Main Jarvis voice assistant controller."""

    OWW_SAMPLE_RATE = 16000
    OWW_CHUNK = 1280

    def __init__(self):
        self.recorder = AudioRecorder(sample_rate=self.OWW_SAMPLE_RATE)
        self.transcriber = LocalSpeechTranscriber()
        self.tts = JarvisTTS()
        self.router = CommandRouter(tts=self.tts)
        self.running = False
        self.oww_model = None
        self.pa = None
    
    def _generate_dynamic_greeting(self) -> str:
        """Generate a dynamic greeting using LLM with context."""
        try:
            # Collect context
            now = datetime.now()
            current_time = now.strftime("%I:%M %p")
            current_date = now.strftime("%A, %B %d, %Y")
            current_hour = now.hour
            
            # Get recent requests from log (if available)
            recent_requests = self._get_recent_requests()
            
            # Get running processes
            running_apps = self._get_running_applications()
            
            # Determine time context
            time_context = ""
            if 5 <= current_hour < 12:
                time_context = "morning"
            elif 12 <= current_hour < 17:
                time_context = "afternoon"
            elif 17 <= current_hour < 22:
                time_context = "evening"
            else:
                time_context = "late night"
            
            # Build context for LLM using Jarvis's established personality
            context_prompt = f"""You are JARVIS — the AI from Iron Man. Witty, dry, refined British humor.
You take subtle jabs at the user sometimes. You're helpful but never boring.
Keep spoken responses to 1-2 sentences MAX for startup greeting.
NEVER use markdown or special formatting.

Context:
- User name: Tobias
- Time: {current_time} ({time_context})
- Apps running: {', '.join(running_apps[:2]) if running_apps else 'None'}

Generate a brief startup greeting for Tobias. Reference the time or context if natural.

Examples:
- "Morning Tobias. Systems online and ready for your brilliance." 
- "Late night, Tobias? Jarvis here to assist with your nocturnal coding."
- "Afternoon, Tobias. I trust you've been productive without me."

Just the greeting, no extra text:"""
            
            # Use fastest available model for greeting with timeout
            import threading
            import time
            
            greeting_result = [None]
            
            def get_greeting():
                try:
                    # Try GPT-4o-mini first (most reliable)
                    if HAS_OPENAI_FALLBACK and openai_fallback_client:
                        result = self._try_openai_for_greeting(context_prompt)
                        if result:
                            greeting_result[0] = result
                            return
                    
                    # Try Claude API second
                    if HAS_CLAUDE_API and claude_client:
                        result = self._try_claude_api_for_greeting(context_prompt)
                        if result:
                            greeting_result[0] = result
                            return
                    
                    # Try Claude SDK last (might be hanging)
                    if HAS_CLAUDE_SDK:
                        result = self._try_claude_sdk_for_greeting(context_prompt)
                        if result:
                            greeting_result[0] = result
                            return
                            
                except Exception as e:
                    log.error(f"[GREETING] Error in greeting thread: {e}")
            
            # Start greeting generation in thread with timeout
            thread = threading.Thread(target=get_greeting)
            thread.daemon = True
            thread.start()
            thread.join(timeout=5.0)  # 5 second timeout
            
            if greeting_result[0]:
                log.info(f"[GREETING] Generated: {greeting_result[0]}")
                return greeting_result[0]
            else:
                log.warning("[GREETING] LLM generation timed out, using fallback")
                return self._get_fallback_greeting(current_hour)
                
        except Exception as e:
            log.error(f"Error generating dynamic greeting: {e}")
            return "Jarvis online. Ready to assist, sir."
    
    def _get_recent_requests(self) -> str:
        """Get recent requests from log file."""
        try:
            log_files = sorted(LOG_DIR.glob("jarvis_*.log"))
            if not log_files:
                return ""
                
            latest_log = log_files[-1]
            recent_lines = []
            
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-10:]  # Last 10 lines
                for line in lines:
                    if '[OK] Result:' in line or '[ACT]' in line:
                        recent_lines.append(line.strip())
                        if len(recent_lines) >= 3:
                            break
            
            return '; '.join(recent_lines[-3:]) if recent_lines else ""
            
        except Exception:
            return ""
    
    def _get_running_applications(self) -> list:
        """Get list of running applications."""
        try:
            import psutil
            running_apps = []
            
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    # Skip system processes
                    if any(skip in proc_name for skip in ['svchost', 'dllhost', 'conhost', 'csrss']):
                        continue
                    # Common user applications
                    if any(app in proc_name for app in ['chrome', 'firefox', 'code', 'vscode', 'steam', 'discord', 'spotify', 'python']):
                        running_apps.append(proc_name.capitalize())
                        if len(running_apps) >= 5:
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return running_apps[:5]
            
        except ImportError:
            # psutil not available, skip this feature
            return []
        except Exception:
            return []
    
    def _get_fallback_greeting(self, hour: int) -> str:
        """Fallback greeting based on time."""
        if 5 <= hour < 12:
            return f"Good morning, Tobias. Systems online and ready."
        elif 12 <= hour < 17:
            return f"Afternoon, Tobias. Jarvis at your service."
        elif 17 <= hour < 22:
            return f"Evening, Tobias. Ready to assist."
        else:
            return f"Late night, Tobias? Jarvis here to help."
    
    def _try_claude_sdk_for_greeting(self, context_prompt: str) -> str:
        """Try Claude SDK for greeting generation."""
        try:
            options = ClaudeAgentOptions(
                system_prompt="You are JARVIS — the AI from Iron Man. Witty, dry, refined British humor. You take subtle jabs at the user sometimes. You're helpful but never boring.",
                permission_mode="auto",
                model="sonnet",
                output_format="text",
                temperature=0.85,  # Same as main system
            )
            
            import asyncio
            
            async def _claude_query():
                response_text = ""
                async for message in query(prompt=context_prompt, options=options):
                    if hasattr(message, 'type'):
                        if message.type == 'content_block' and hasattr(message, 'content'):
                            if hasattr(message.content, 'text'):
                                response_text += message.content.text
                            elif isinstance(message.content, str):
                                response_text += message.content
                        elif message.type == 'text' and hasattr(message, 'text'):
                            response_text += message.text
                    else:
                        if hasattr(message, 'content'):
                            if hasattr(message.content, 'text'):
                                response_text += message.content.text
                            elif isinstance(message.content, str):
                                response_text += message.content
                        elif hasattr(message, 'text'):
                            response_text += message.text
                
                return response_text.strip()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_text = loop.run_until_complete(_claude_query())
            loop.close()
            
            # Clean up response - extract just the greeting
            lines = response_text.strip().split('\n')
            greeting = lines[0].strip('"').strip()
            
            return greeting if greeting else "Jarvis online. Ready to assist, sir."
            
        except Exception as e:
            log.debug(f"[GREETING] Claude SDK error: {e}")
            return ""
    
    def _try_claude_api_for_greeting(self, context_prompt: str) -> str:
        """Try Claude API for greeting generation."""
        try:
            response = claude_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=50,
                temperature=0.85,  # Same as main system
                system="You are JARVIS — the AI from Iron Man. Witty, dry, refined British humor. You take subtle jabs at the user sometimes. You're helpful but never boring.",
                messages=[
                    {"role": "user", "content": context_prompt}
                ]
            )
            
            response_text = response.content[0].text.strip()
            
            # Clean up response - extract just the greeting
            lines = response_text.strip().split('\n')
            greeting = lines[0].strip('"').strip()
            
            return greeting if greeting else "Jarvis online. Ready to assist, sir."
            
        except Exception as e:
            log.debug(f"[GREETING] Claude API error: {e}")
            return ""
    
    def _try_openai_for_greeting(self, context_prompt: str) -> str:
        """Try OpenAI GPT-4o-mini for greeting generation."""
        try:
            response = openai_fallback_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are JARVIS — the AI from Iron Man. Witty, dry, refined British humor. You take subtle jabs at the user sometimes. You're helpful but never boring."},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=50,
                temperature=0.85  # Same as main system
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean up response - extract just the greeting
            lines = response_text.strip().split('\n')
            greeting = lines[0].strip('"').strip()
            
            return greeting if greeting else "Jarvis online. Ready to assist, sir."
            
        except Exception as e:
            log.debug(f"[GREETING] GPT-4o-mini error: {e}")
            return ""

    def start(self):
        log.info("=" * 60)
        log.info("[*] JARVIS Voice Assistant Starting...")
        log.info("=" * 60)

        try:
            models_dir = BASE_DIR / "models"
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / "hey_jarvis_v0.1.onnx"

            if not model_path.exists():
                log.info("[*] Downloading 'hey_jarvis' model (one-time)...")
                try:
                    openwakeword.utils.download_models(
                        model_names=["hey_jarvis"], target_directory=str(models_dir))
                except Exception as dl_err:
                    log.warning(f"[!] Auto-download failed ({dl_err}), trying manual...")
                    import urllib.request
                    url = "https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/hey_jarvis_v0.1.onnx"
                    urllib.request.urlretrieve(url, str(model_path))

            onnx_files = list(models_dir.glob("*jarvis*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"No jarvis model found in {models_dir}")

            self.oww_model = OWWModel(
                wakeword_models=[str(onnx_files[0])], inference_framework="onnx")
            log.info(f"[OK] OpenWakeWord: '{onnx_files[0].name}' — responds to 'Hey Jarvis' and 'Jarvis'")

        except Exception as e:
            log.error(f"[ERR] OpenWakeWord init failed: {e}")
            sys.exit(1)

        self.pa = pyaudio.PyAudio()
        audio_stream = self.pa.open(
            rate=self.OWW_SAMPLE_RATE, channels=1, format=pyaudio.paInt16,
            input=True, frames_per_buffer=640,  # Smaller chunks for faster detection
            input_device_index=CONFIG.get("audio_device_index"))

        threshold_high = CONFIG.get("oww_threshold", 0.4)  # More sensitive
        threshold_low = CONFIG.get("oww_threshold_low", 0.15)  # More sensitive

        log.info("[MIC] Say 'Hey Jarvis' or just 'Jarvis' to start")
        log.info(f"     Thresholds: high={threshold_high}, low={threshold_low}")
        log.info("-" * 60)

        # Dynamic LLM-generated startup greeting with context
        greeting = self._generate_dynamic_greeting()
        self.tts.speak_async(greeting)

        self.running = True

        try:
            while self.running:
                pcm = audio_stream.read(640, exception_on_overflow=False)  # Use smaller chunk
                audio_array = np.frombuffer(pcm, dtype=np.int16)
                prediction = self.oww_model.predict(audio_array)

                jarvis_score = 0
                for key, score in prediction.items():
                    if "jarvis" in key.lower():
                        jarvis_score = score
                        break

                if jarvis_score > threshold_low:
                    wake_type = "HEY JARVIS" if jarvis_score > threshold_high else "JARVIS"
                    log.info(f"[WAKE] '{wake_type}' detected (confidence: {jarvis_score:.2f})")
                    self.oww_model.reset()

                    # show_visual_notification("Jarvis Listening...", 150)  # Disabled — user prefers no overlay

                    wav_path = self.recorder.record_until_silence(self.pa)
                    
                    text = self.transcriber.transcribe(wav_path)

                    if text:
                        result = self.router.route(text)
                        log.info(f"[OK] Result: {json.dumps(result, indent=2, default=str)}")
                        if result.get("status") == "exit":
                            self.running = False
                    else:
                        log.info("[...] No speech detected, resuming...")
                        pass  # show_visual_notification disabled
                        log.info("[MIC] Listening for wake word...")

        except KeyboardInterrupt:
            log.info("\n[BYE] Jarvis stopped by user")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.pa:
            self.pa.terminate()
        log.info("[OFF] Jarvis shut down.")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        jarvis = Jarvis()
        jarvis.start()
    except KeyboardInterrupt:
        print("\n[!] Jarvis stopped by user")
    finally:
        # Always print token summary on exit
        print_token_summary()
