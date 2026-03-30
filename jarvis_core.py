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
from pathlib import Path
from datetime import datetime

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

# OpenAI client for Whisper (still needed for transcription)
whisper_client = None
if CONFIG.get("openai_api_key"):
    try:
        from openai import OpenAI
        whisper_client = OpenAI(api_key=CONFIG["openai_api_key"])
    except ImportError:
        print("[!] openai package not found for Whisper. Run: pip install openai")
        whisper_client = None
else:
    whisper_client = None

# OpenAI client for GPT-4o-mini fallback (uses existing key)
if HAS_OPENAI_FALLBACK and CONFIG.get("openai_api_key"):
    try:
        openai_fallback_client = openai.OpenAI(api_key=CONFIG["openai_api_key"])
        print("[OK] OpenAI GPT-4o-mini initialized (fallback)")
    except Exception as e:
        print(f"[!] Failed to initialize OpenAI fallback: {e}")
        openai_fallback_client = None
elif HAS_OPENAI_FALLBACK:
    print("[!] OpenAI API key not configured for fallback")

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
        time.sleep(0.5)

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

class WhisperTranscriber:
    """Transcribes audio using OpenAI Whisper API."""

    def __init__(self):
        self.model = CONFIG.get("whisper_model", "whisper-1")

    def transcribe(self, audio_path: Path) -> str:
        """Send audio to Whisper API, return transcription text."""
        log.info(f"[AI] Transcribing with {self.model}...")
        if not whisper_client:
            log.error("Whisper client not available")
            return ""
        try:
            with open(audio_path, "rb") as audio_file:
                response = whisper_client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language="en"
                )
            text = response.text.strip()
            log.info(f"[TXT] Transcription: \"{text}\"")
            return text
        except Exception as e:
            log.error(f"Whisper API error: {e}")
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

=== IMPORTANT RULES ===
- For time/date questions: use the time_hint provided, don't say you can't tell time
- For opening apps/sites: use [ACTION: chain "launch ..."]
- For keyboard shortcuts: use [ACTION: chain "key ..."]
- For complex multi-step PC tasks: chain multiple commands with semicolons and waits
- For coding tasks or anything needing Claude's intelligence: use [ACTION: claude_code "..."]
- For pure conversation with no PC action needed: use [ACTION: speak_only]
- ALWAYS include [SPEAK] — you must always respond verbally
- Never explain the action format to the user — just do it and talk naturally
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
        # For voice-based command creation (multi-step state)
        self._creating_command = False
        self._new_cmd_data = {}

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
            self._run_cc("chain", chain_cmd)
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
            "manage commands", "open manager"]):
            gui_path = BASE_DIR / "jarvis_gui.py"
            subprocess.Popen([sys.executable, str(gui_path)],
                            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
            if self.tts:
                self.tts.speak_async("Opening the command manager.")
            return {"status": "ok", "action": "open_command_manager"}

        if any(text_lower.startswith(p) for p in [
            "add command", "new command", "create command",
            "add a command", "create a command"]):
            return self._start_voice_create(text_lower)

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

        # ── Everything else → Claude Brain ──
        log.info("[BRAIN] Sending to Claude for interpretation...")
        self._ask_claude(text)
        return {"status": "ok", "action": "claude_brain", "prompt": text}

    def _ask_claude(self, user_text: str):
        """Send user speech to Claude SDK with Gemini fallback. Runs in background thread."""
        def _run():
            try:
                now = datetime.now()
                time_hint = now.strftime("%I:%M %p on %A, %B %d, %Y")
                system_prompt = JARVIS_SYSTEM_PROMPT.replace("{time_hint}", time_hint)
                system_prompt += f"\n\nCurrent time: {time_hint}\n\nUser said: {user_text}"

                # Try Claude SDK first (primary) - with retry logic
                if HAS_CLAUDE_SDK:
                    log.info(f"[BRAIN] Trying Claude SDK...")
                    for attempt in range(2):  # Try twice
                        response_text = self._try_claude_sdk(user_text, system_prompt)
                        if response_text:
                            self._handle_response(response_text)
                            return
                        elif attempt == 0:
                            log.info(f"[BRAIN] Claude SDK attempt 1 failed, retrying...")
                            time.sleep(1)  # Brief delay before retry
                    log.warning("[BRAIN] Claude SDK failed after 2 attempts, trying fallback...")
                
                # Fallback to OpenAI GPT-4o-mini (cheap, uses existing key)
                if HAS_OPENAI_FALLBACK and openai_fallback_client:
                    log.info(f"[BRAIN] Using GPT-4o-mini fallback...")
                    response_text = self._try_openai_fallback(user_text, system_prompt)
                    if response_text:
                        self._handle_response(response_text)
                        return
                    else:
                        log.warning("[BRAIN] GPT-4o-mini fallback failed")
                
                # Nothing worked
                log.error("[BRAIN] All AI methods failed")
                if self.tts:
                    self.tts.speak("I'm having trouble connecting right now, sir.")

            except Exception as e:
                log.error(f"[BRAIN] Error: {e}")
                if self.tts:
                    self.tts.speak("Something went wrong on my end. Apologies, sir.")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
    
    def _try_claude_sdk(self, user_text: str, system_prompt: str) -> str:
        """Try Claude Code Agent SDK (primary method)."""
        try:
            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                permission_mode="auto",
                model="sonnet",
                output_format="text",
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
            
            return response_text
            
        except Exception as e:
            log.debug(f"[BRAIN] Claude SDK error: {e}")
            return ""
    
    def _try_openai_fallback(self, user_text: str, system_prompt: str) -> str:
        """Try OpenAI GPT-4o-mini (cheap fallback)."""
        try:
            response = openai_fallback_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            log.debug(f"[BRAIN] GPT-4o-mini error: {e}")
            return ""
    
    def _handle_response(self, response_text: str):
        """Process AI response and handle actions/speech."""
        if response_text:
            log.info(f"[BRAIN] Response: {response_text[:300]}...")
            speak_text = self._parse_and_execute(response_text)
            if speak_text and self.tts:
                self.tts.speak(speak_text)
        else:
            log.warning("[BRAIN] Empty response")
            if self.tts:
                self.tts.speak("I seem to have drawn a blank on that one. Could you try again?")

    # ─── Voice Command Creation ──────────────────────────────────────────

    def _start_voice_create(self, text: str):
        log.info("[+] Starting voice command creation...")
        self._creating_command = True
        self._new_cmd_data = {"step": "name"}
        if self.tts:
            self.tts.speak_async("Alright, let's create a new command. What shall we call it?")
        return {"status": "listening", "action": "voice_create", "message": "Say the command name"}

    def _voice_create_step(self, text: str):
        step = self._new_cmd_data.get("step", "name")

        if text.lower().strip() in ["cancel", "stop", "nevermind", "never mind"]:
            self._creating_command = False
            self._new_cmd_data = {}
            if self.tts:
                self.tts.speak_async("Command creation cancelled.")
            return {"status": "cancelled", "action": "voice_create"}

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


# ─── Sound Feedback ──────────────────────────────────────────────────────────

def play_beep(frequency=440, duration_ms=200):
    if not CONFIG.get("sound_feedback", True):
        return
    try:
        import winsound
        winsound.Beep(frequency, duration_ms)
    except Exception:
        print("\a", end="", flush=True)


# ─── Text-to-Speech Engine ──────────────────────────────────────────────────

class JarvisTTS:
    """Neural TTS via Microsoft Edge (free). Smooth British male voice."""

    VOICE = "en-GB-RyanNeural"

    def __init__(self):
        self._lock = threading.Lock()
        self._tts_dir = BASE_DIR / "tts_cache"
        self._tts_dir.mkdir(exist_ok=True)
        self.available = HAS_EDGE_TTS
        
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

        log.info(f"[TTS] Speaking: {clean[:100]}...")
        with self._lock:
            try:
                # Use timestamp to avoid file conflicts
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                audio_path = self._tts_dir / f"jarvis_response_{timestamp}.mp3"

                async def _generate():
                    communicate = edge_tts.Communicate(clean, self.VOICE, rate="-5%")
                    await communicate.save(str(audio_path))

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_generate())
                loop.close()

                # Fast pygame playback instead of PowerShell
                if HAS_PYGAME and audio_path.exists():
                    try:
                        pygame.mixer.music.load(str(audio_path))
                        pygame.mixer.music.play()
                        
                        # Wait for playback to finish
                        while pygame.mixer.music.get_busy():
                            pygame.time.wait(100)
                            
                    except Exception as e:
                        log.warning(f"[TTS] Pygame playback failed, falling back: {e}")
                        # Fallback to original method
                        self._fallback_playback(str(audio_path))
                else:
                    # Original fallback method
                    self._fallback_playback(str(audio_path))
                
                # Clean up old audio files (keep only last 5)
                try:
                    audio_files = sorted(self._tts_dir.glob("jarvis_response_*.mp3"))
                    if len(audio_files) > 5:
                        for old_file in audio_files[:-5]:
                            try:
                                old_file.unlink()
                            except Exception as cleanup_e:
                                log.debug(f"[TTS] Cleanup failed for {old_file}: {cleanup_e}")
                except Exception as cleanup_e:
                    log.debug(f"[TTS] Cleanup error: {cleanup_e}")

            except Exception as e:
                log.error(f"[TTS] Speech error: {e}")
    
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
        self.transcriber = WhisperTranscriber()
        self.tts = JarvisTTS()
        self.router = CommandRouter(tts=self.tts)
        self.running = False
        self.oww_model = None
        self.pa = None

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
            input=True, frames_per_buffer=self.OWW_CHUNK,
            input_device_index=CONFIG.get("audio_device_index"))

        threshold_high = CONFIG.get("oww_threshold", 0.5)
        threshold_low = CONFIG.get("oww_threshold_low", 0.2)

        log.info("[MIC] Say 'Hey Jarvis' or just 'Jarvis' to start")
        log.info(f"     Thresholds: high={threshold_high}, low={threshold_low}")
        log.info("-" * 60)

        # Random startup greeting
        greeting = random.choice(STARTUP_GREETINGS)
        self.tts.speak_async(greeting)

        self.running = True

        try:
            while self.running:
                pcm = audio_stream.read(self.OWW_CHUNK, exception_on_overflow=False)
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

                    play_beep(880, 150)

                    wav_path = self.recorder.record_until_silence(self.pa)
                    play_beep(440, 100)

                    text = self.transcriber.transcribe(wav_path)

                    if text:
                        result = self.router.route(text)
                        log.info(f"[OK] Result: {json.dumps(result, indent=2, default=str)}")
                        if result.get("status") == "exit":
                            self.running = False
                    else:
                        log.info("[...] No speech detected, resuming...")

                    play_beep(660, 100)
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
    jarvis = Jarvis()
    jarvis.start()
