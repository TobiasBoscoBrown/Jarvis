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
from pathlib import Path
from datetime import datetime

# OpenWakeWord (free, open-source wake word detection)
try:
    import openwakeword
    from openwakeword.model import Model as OWWModel
except ImportError:
    print("[!] openwakeword not found. Run: pip install openwakeword")
    sys.exit(1)

# Optional: openai for Whisper API
try:
    from openai import OpenAI
except ImportError:
    print("[!] openai package not found. Run: pip install openai")
    sys.exit(1)

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

# OpenAI client
client = OpenAI(api_key=CONFIG["openai_api_key"])

# ─── Audio Recording ─────────────────────────────────────────────────────────

class AudioRecorder:
    """Records audio after wake-word until silence is detected."""

    def __init__(self, sample_rate=16000, channels=1, chunk_size=512):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.silence_threshold = CONFIG.get("silence_threshold", 1.5)
        self.energy_threshold = 500  # RMS energy below which = silence

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
        time.sleep(0.3)

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
        try:
            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
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


# ─── Natural Language → cc.py Chain Parser ───────────────────────────────────

import re

# Word-to-number map for spoken numbers
WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "fifty": 50, "hundred": 100,
    "once": 1, "twice": 2, "thrice": 3,
}

# Key name aliases (what people say → cc.py key name)
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


# ─── Command Router ──────────────────────────────────────────────────────────

class CommandRouter:
    """Routes transcribed voice commands to appropriate handlers."""

    def __init__(self):
        self.custom_commands = self._load_custom_commands()
        self.cc_py = CONFIG.get("cc_py_path", r"C:\Users\tobia\Desktop\ClaudeBridge\skills\cc.py")
        self._commands_path = BASE_DIR / CONFIG.get("custom_commands_file", "commands/custom_commands.json")
        self._commands_mtime = self._get_mtime()
        self.handlers = {
            "open_cowork": self.open_cowork,
            "open_terminal": self.open_terminal,
            "screenshot": self.take_screenshot,
            "open_cowork_convo": self.open_cowork_conversation,
            "prompt_cowork": self.prompt_cowork,
            "prompt_claude_code": self.prompt_claude_code,
            "type_text": self.type_text,
            "search_web": self.search_web,
            "focus_window": self.focus_window,
        }
        # For voice-based command creation (multi-step state)
        self._creating_command = False
        self._new_cmd_data = {}

    def _get_mtime(self):
        try:
            return self._commands_path.stat().st_mtime
        except Exception:
            return 0

    def _hot_reload_if_changed(self):
        """Auto-reload commands if the JSON file was modified on disk."""
        current_mtime = self._get_mtime()
        if current_mtime != self._commands_mtime:
            self._commands_mtime = current_mtime
            self.custom_commands = self._load_custom_commands()
            log.info("[HOT] Commands auto-reloaded (file changed on disk)")

    def _load_custom_commands(self):
        if self._commands_path.exists() if hasattr(self, '_commands_path') else False:
            custom_path = self._commands_path
        else:
            custom_path = BASE_DIR / CONFIG.get("custom_commands_file", "commands/custom_commands.json")
        if custom_path.exists():
            with open(custom_path, "r") as f:
                return json.load(f)
        return {"commands": [], "aliases": {}}

    def reload_commands(self):
        """Hot-reload custom commands from disk."""
        self.custom_commands = self._load_custom_commands()
        self._commands_mtime = self._get_mtime()
        log.info("[OK] Custom commands reloaded")

    def _run_cc(self, *args):
        """Run cc.py via cmd.exe shell to preserve quoted chain arguments."""
        # Build command string — quote the chain arg so cmd.exe keeps it as one piece
        arg_parts = []
        for a in args:
            if " " in a or ";" in a:
                arg_parts.append(f'"{a}"')
            else:
                arg_parts.append(a)
        cmd_str = f'"{sys.executable}" "{self.cc_py}" {" ".join(arg_parts)}'
        log.info(f"[CMD] Running: cc.py {' '.join(arg_parts)}")
        try:
            result = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=CONFIG.get("command_timeout", 30)
            )
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

    def route(self, text: str) -> dict:
        """Parse transcribed text and route to the right handler."""
        if not text:
            return {"status": "empty", "message": "No speech detected"}

        # Auto hot-reload: check if commands file changed on disk
        self._hot_reload_if_changed()

        text_lower = text.lower().strip()
        log.info(f"[>>] Routing command: \"{text_lower}\"")

        # ── Voice command creation flow (multi-step) ──
        if self._creating_command:
            return self._voice_create_step(text)

        # ── Meta commands ──
        if any(text_lower.startswith(p) for p in ["reload commands", "refresh commands"]):
            self.reload_commands()
            return {"status": "ok", "action": "reload_commands"}

        # Open the GUI command manager
        if any(text_lower.startswith(p) for p in [
            "open command manager", "command manager", "open commands",
            "manage commands", "open manager", "launch command manager"]):
            return self._open_command_manager()

        # Voice command creation
        if any(text_lower.startswith(p) for p in [
            "add command", "new command", "create command",
            "add a command", "create a command", "new voice command"]):
            return self._start_voice_create(text_lower)

        if any(text_lower.startswith(p) for p in ["list commands", "show commands", "what commands"]):
            return self._list_commands()

        # ── Cowork conversation commands ──
        # "open cowork convo [name]" or "open claude cowork conversation [name]"
        for prefix in ["open cowork convo ", "open cowork conversation ",
                       "open claude cowork convo ", "open claude cowork conversation ",
                       "open cowork convo of ", "open cowork conversation of "]:
            if text_lower.startswith(prefix):
                convo_name = text_lower[len(prefix):].strip()
                return self.open_cowork_conversation(convo_name)

        # ── Prompt Cowork ──
        for prefix in ["prompt cowork ", "tell cowork ", "prompt claude cowork ",
                       "ask cowork ", "cowork prompt ", "prompt cowork to ",
                       "tell cowork to ", "ask cowork to "]:
            if text_lower.startswith(prefix):
                prompt_text = text[len(prefix):].strip()  # Keep original casing
                return self.prompt_cowork(prompt_text)

        # ── Prompt Claude Code ──
        for prefix in ["prompt claude code ", "tell claude code ", "ask claude code ",
                       "claude code ", "prompt claude code to ",
                       "tell claude code to ", "ask claude code to ",
                       "run in terminal ", "terminal command "]:
            if text_lower.startswith(prefix):
                prompt_text = text[len(prefix):].strip()
                return self.prompt_claude_code(prompt_text)

        # ── Focus window ──
        for prefix in ["focus ", "switch to ", "go to ", "bring up "]:
            if text_lower.startswith(prefix):
                window_name = text_lower[len(prefix):].strip()
                # Skip if it's a more specific command
                if not any(window_name.startswith(w) for w in ["cowork", "claude", "terminal"]):
                    return self.focus_window(window_name)

        # ── Search web ──
        for prefix in ["search for ", "google ", "search ", "look up "]:
            if text_lower.startswith(prefix):
                query = text[len(prefix):].strip()
                return self.search_web(query)

        # ── Type/dictate ──
        for prefix in ["type ", "dictate ", "write "]:
            if text_lower.startswith(prefix):
                content = text[len(prefix):].strip()
                return self.type_text(content)

        # ── Check custom commands ──
        for cmd in self.custom_commands.get("commands", []):
            for trigger in cmd.get("triggers", []):
                if text_lower.startswith(trigger.lower()):
                    action = cmd.get("action", "")
                    if action in self.handlers:
                        remaining = text_lower[len(trigger):].strip()
                        return self.handlers[action](remaining) if remaining else self.handlers[action]()
                    elif action.startswith("chain:"):
                        chain_cmd = action[6:]
                        return self._run_cc("chain", f'"{chain_cmd}"')

        # ── Simple built-in triggers ──
        if text_lower in ["open cowork", "open claude cowork", "launch cowork"]:
            return self.open_cowork()
        if text_lower in ["open terminal", "open command prompt", "launch terminal"]:
            return self.open_terminal()
        if text_lower in ["screenshot", "take screenshot", "capture screen"]:
            return self.take_screenshot()
        if text_lower in ["stop", "quit", "exit", "goodbye", "shut down"]:
            return {"status": "exit", "message": "Shutting down Jarvis"}

        # ── Hot commands: natural language → keyboard/mouse actions ──
        if looks_like_hot_command(text):
            chain = parse_natural_to_chain(text)
            if chain:
                log.info(f"[HOT] Parsed chain: {chain}")
                return self._run_cc("chain", chain)

        # ── Fallback: treat as a prompt to Cowork ──
        log.info("[?] No specific command matched — sending as Cowork prompt")
        return self.prompt_cowork(text)

    # ─── Handler Methods ─────────────────────────────────────────────────

    def open_cowork(self, *args):
        """Open Claude Cowork in browser."""
        log.info("[WEB] Opening Claude Cowork...")
        return self._run_cc("chain",
            "launch https://claude.ai; wait 3; screenshot")

    def open_terminal(self, *args):
        """Open Windows Terminal."""
        log.info("[SYS] Opening Terminal...")
        return self._run_cc("chain",
            "launch wt.exe; wait 2; screenshot")

    def take_screenshot(self, *args):
        """Take a screenshot."""
        log.info("[CAM] Taking screenshot...")
        return self._run_cc("screenshot")

    def open_cowork_conversation(self, convo_name: str):
        """Open a specific Cowork conversation by name."""
        log.info(f"[NAV] Opening Cowork conversation: {convo_name}")
        return self._run_cc("chain",
            f"launch https://claude.ai; wait 3; "
            f"click_text Search --window Chrome; wait 0.5; "
            f"type {convo_name}; wait 1; screenshot")

    def prompt_cowork(self, prompt_text: str):
        """Type a prompt into Claude Cowork."""
        log.info(f"[>>] Prompting Cowork: {prompt_text[:80]}...")
        return self._run_cc("chain",
            f"focus Claude; wait 0.5; "
            f"click 960 900; wait 0.3; "
            f"type {prompt_text}; wait 0.3; "
            f"key enter; wait 1; screenshot")

    def prompt_claude_code(self, prompt_text: str):
        """Send a command to Claude Code in the terminal."""
        log.info(f"[>>] Prompting Claude Code: {prompt_text[:80]}...")
        terminal_title = CONFIG.get("claude_code_terminal", "Windows Terminal")
        return self._run_cc("chain",
            f"focus {terminal_title}; wait 0.5; "
            f"type {prompt_text}; wait 0.3; "
            f"key enter; wait 1; screenshot")

    def type_text(self, content: str):
        """Type text at current cursor position (voice-to-type)."""
        log.info(f"[KEY] Typing: {content[:60]}...")
        return self._run_cc("type", content)

    def search_web(self, query: str):
        """Open browser and search for something."""
        log.info(f"[WEB] Searching: {query}")
        return self._run_cc("chain",
            f"focus Chrome; wait 0.5; key ctrl+l; wait 0.2; "
            f"type https://www.google.com/search?q={query}; "
            f"key enter; wait 3; screenshot")

    def focus_window(self, window_name: str):
        """Focus/bring a window to foreground."""
        log.info(f"[WIN] Focusing: {window_name}")
        return self._run_cc("focus", window_name)

    # ─── Custom Command Management ──────────────────────────────────────

    def _open_command_manager(self):
        """Launch the GUI command manager."""
        log.info("[GUI] Opening Command Manager...")
        gui_path = BASE_DIR / "jarvis_gui.py"
        subprocess.Popen([sys.executable, str(gui_path)],
                        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
        return {"status": "ok", "action": "open_command_manager"}

    def _start_voice_create(self, text: str):
        """Start the multi-step voice command creation flow."""
        log.info("[+] Starting voice command creation...")
        log.info("    Say the COMMAND NAME after the beep")
        self._creating_command = True
        self._new_cmd_data = {"step": "name"}
        return {
            "status": "listening",
            "action": "voice_create",
            "message": "Say the command name now"
        }

    def _voice_create_step(self, text: str):
        """Handle each step of voice command creation."""
        step = self._new_cmd_data.get("step", "name")

        if text.lower().strip() in ["cancel", "stop", "nevermind", "never mind"]:
            self._creating_command = False
            self._new_cmd_data = {}
            log.info("[X] Command creation cancelled")
            return {"status": "cancelled", "action": "voice_create"}

        if step == "name":
            self._new_cmd_data["name"] = text.strip()
            self._new_cmd_data["step"] = "trigger"
            log.info(f"[+] Name: '{text.strip()}'")
            log.info("    Now say the TRIGGER PHRASE (what you'll say to activate it)")
            return {"status": "listening", "step": "trigger",
                    "message": f"Name set to '{text.strip()}'. Now say the trigger phrase."}

        elif step == "trigger":
            self._new_cmd_data["trigger"] = text.strip().lower()
            self._new_cmd_data["step"] = "action"
            log.info(f"[+] Trigger: '{text.strip()}'")
            log.info("    Now say the ACTION: 'prompt cowork', 'prompt claude code',")
            log.info("    'open cowork', 'screenshot', 'type text', or 'chain' + description")
            return {"status": "listening", "step": "action",
                    "message": f"Trigger set to '{text.strip()}'. Now say the action type."}

        elif step == "action":
            action_text = text.strip().lower()
            # Map spoken action to handler name
            action_map = {
                "prompt cowork": "prompt_cowork",
                "prompt claude code": "prompt_claude_code",
                "open cowork": "open_cowork",
                "open terminal": "open_terminal",
                "screenshot": "screenshot",
                "type text": "type_text",
                "type": "type_text",
                "search": "search_web",
                "search web": "search_web",
                "focus window": "focus_window",
                "focus": "focus_window",
            }

            action = action_map.get(action_text, action_text)
            # If it starts with "chain", treat the rest as a chain command
            if action_text.startswith("chain"):
                chain_desc = action_text[5:].strip()
                if chain_desc:
                    action = f"chain:{chain_desc}"
                else:
                    action = "chain:"

            # Build and save the command
            new_cmd = {
                "name": self._new_cmd_data["name"],
                "triggers": [self._new_cmd_data["trigger"]],
                "action": action,
                "description": f"Voice-created command"
            }

            self.custom_commands.setdefault("commands", []).append(new_cmd)
            commands_path = BASE_DIR / CONFIG.get("custom_commands_file", "commands/custom_commands.json")
            with open(commands_path, "w") as f:
                json.dump(self.custom_commands, f, indent=4)

            self._creating_command = False
            self._new_cmd_data = {}

            log.info(f"[OK] Command created! Name: '{new_cmd['name']}', "
                     f"Trigger: '{new_cmd['triggers'][0]}', Action: '{action}'")
            log.info("     Command is live immediately - no restart needed!")

            return {"status": "ok", "action": "voice_create_complete", "command": new_cmd}

        # Shouldn't get here
        self._creating_command = False
        return {"status": "error", "message": "Unknown creation step"}

    def _list_commands(self):
        """List all available commands."""
        built_in = [
            "open cowork", "open terminal", "screenshot",
            "open cowork convo [name]", "prompt cowork [text]",
            "prompt claude code [text]", "type [text]",
            "search [query]", "focus [window]",
            "open command manager", "add command (voice)",
            "reload commands", "list commands",
            "stop / quit / exit"
        ]
        custom = [c["name"] for c in self.custom_commands.get("commands", [])]
        result = {
            "status": "ok",
            "built_in_commands": built_in,
            "custom_commands": custom,
            "total": len(built_in) + len(custom)
        }
        log.info(f"[LIST] Commands: {json.dumps(result, indent=2)}")
        return result


# ─── Sound Feedback ──────────────────────────────────────────────────────────

def play_beep(frequency=440, duration_ms=200):
    """Play a quick beep to indicate wake-word detected."""
    if not CONFIG.get("sound_feedback", True):
        return
    try:
        import winsound
        winsound.Beep(frequency, duration_ms)
    except Exception:
        # Fallback: print bell character
        print("\a", end="", flush=True)


# ─── Main Loop ───────────────────────────────────────────────────────────────

class Jarvis:
    """Main Jarvis voice assistant controller."""

    # OpenWakeWord settings
    OWW_SAMPLE_RATE = 16000
    OWW_CHUNK = 1280  # 80ms at 16kHz — OpenWakeWord's expected frame size

    def __init__(self):
        self.recorder = AudioRecorder(sample_rate=self.OWW_SAMPLE_RATE)
        self.transcriber = WhisperTranscriber()
        self.router = CommandRouter()
        self.running = False
        self.oww_model = None
        self.pa = None

    def start(self):
        """Initialize OpenWakeWord and start listening."""
        log.info("=" * 60)
        log.info("[*] JARVIS Voice Assistant Starting...")
        log.info("=" * 60)

        try:
            # Store models in our own folder to avoid system permission issues
            models_dir = BASE_DIR / "models"
            models_dir.mkdir(exist_ok=True)

            model_path = models_dir / "hey_jarvis_v0.1.onnx"

            # Download the hey_jarvis model if we don't have it yet
            if not model_path.exists():
                log.info("[*] Downloading 'hey_jarvis' model (one-time, ~few MB)...")
                try:
                    openwakeword.utils.download_models(
                        model_names=["hey_jarvis"],
                        target_directory=str(models_dir)
                    )
                    log.info("[OK] Model downloaded to models/ folder")
                except Exception as dl_err:
                    log.warning(f"[!] Auto-download failed ({dl_err}), trying manual download...")
                    # Fallback: download directly with urllib
                    import urllib.request
                    url = "https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/hey_jarvis_v0.1.onnx"
                    urllib.request.urlretrieve(url, str(model_path))
                    log.info("[OK] Model downloaded manually to models/ folder")

            # Find the .onnx model file
            onnx_files = list(models_dir.glob("*jarvis*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"No jarvis model found in {models_dir}")

            # Initialize OpenWakeWord with our local model path
            self.oww_model = OWWModel(
                wakeword_models=[str(onnx_files[0])],
                inference_framework="onnx"
            )
            threshold = CONFIG.get("oww_threshold", 0.5)
            log.info(f"[OK] OpenWakeWord initialized (model: '{onnx_files[0].name}', threshold: {threshold})")
            log.info(f"     100% free - no API keys needed for wake word detection")

        except Exception as e:
            log.error(f"[ERR] OpenWakeWord init failed: {e}")
            log.error("      Try: pip install openwakeword")
            sys.exit(1)

        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        device_index = CONFIG.get("audio_device_index")

        audio_stream = self.pa.open(
            rate=self.OWW_SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.OWW_CHUNK,
            input_device_index=device_index
        )

        threshold = CONFIG.get("oww_threshold", 0.5)

        log.info("[MIC] Microphone active — say 'Hey Jarvis' to start a command")
        log.info("   Press Ctrl+C to quit")
        log.info("-" * 60)

        self.running = True

        try:
            while self.running:
                # Read audio frame for wake-word detection
                pcm = audio_stream.read(self.OWW_CHUNK, exception_on_overflow=False)
                audio_array = np.frombuffer(pcm, dtype=np.int16)

                # Feed to OpenWakeWord — predict() returns a dict of {model_name: score}
                prediction = self.oww_model.predict(audio_array)

                # Get the score for hey_jarvis from the prediction dict
                # The key might be the full path or just the model name
                jarvis_score = 0
                for key, score in prediction.items():
                    if "jarvis" in key.lower():
                        jarvis_score = score
                        break

                if jarvis_score > threshold:
                    log.info(f"[WAKE] Wake word detected! — 'HEY JARVIS' (confidence: {jarvis_score:.2f})")
                    # Reset the model state to avoid repeat triggers
                    self.oww_model.reset()

                    play_beep(880, 150)  # High beep = listening

                    # Record the command
                    wav_path = self.recorder.record_until_silence(self.pa)

                    play_beep(440, 100)  # Low beep = processing

                    # Transcribe
                    text = self.transcriber.transcribe(wav_path)

                    if text:
                        # Route command
                        result = self.router.route(text)
                        log.info(f"[OK] Result: {json.dumps(result, indent=2, default=str)}")

                        if result.get("status") == "exit":
                            log.info("[BYE] Jarvis shutting down...")
                            self.running = False
                    else:
                        log.info("[...] No speech detected, resuming listening...")

                    play_beep(660, 100)  # Mid beep = ready again
                    log.info("[MIC] Listening for wake word...")

        except KeyboardInterrupt:
            log.info("\n[BYE] Jarvis stopped by user (Ctrl+C)")
        finally:
            self.stop()

    def stop(self):
        """Clean up resources."""
        self.running = False
        if self.pa:
            self.pa.terminate()
        log.info("[OFF] Jarvis shut down cleanly.")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    jarvis = Jarvis()
    jarvis.start()
