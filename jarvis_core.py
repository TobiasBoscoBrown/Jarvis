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


# ─── Command Router ──────────────────────────────────────────────────────────

class CommandRouter:
    """Routes transcribed voice commands to appropriate handlers."""

    def __init__(self):
        self.custom_commands = self._load_custom_commands()
        self.cc_py = CONFIG.get("cc_py_path", r"C:\Users\tobia\Desktop\ClaudeBridge\skills\cc.py")
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

    def _load_custom_commands(self):
        custom_path = BASE_DIR / CONFIG.get("custom_commands_file", "commands/custom_commands.json")
        if custom_path.exists():
            with open(custom_path, "r") as f:
                return json.load(f)
        return {"commands": [], "aliases": {}}

    def reload_commands(self):
        """Hot-reload custom commands from disk."""
        self.custom_commands = self._load_custom_commands()
        log.info("[OK] Custom commands reloaded")

    def _run_cc(self, *args):
        """Run cc.py with given arguments via PowerShell."""
        cmd_parts = ['python', f'"{self.cc_py}"'] + list(args)
        cmd_str = ' '.join(cmd_parts)
        full_cmd = f"powershell -Command \"{cmd_str}\""
        log.info(f"[CMD] Running: {cmd_str}")
        try:
            result = subprocess.run(
                full_cmd,
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

        text_lower = text.lower().strip()
        log.info(f"[>>] Routing command: \"{text_lower}\"")

        # ── Meta commands ──
        if any(text_lower.startswith(p) for p in ["reload commands", "refresh commands"]):
            self.reload_commands()
            return {"status": "ok", "action": "reload_commands"}

        if any(text_lower.startswith(p) for p in ["add command", "new command", "create command"]):
            return self._handle_add_command(text_lower)

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

        # ── Fallback: treat as a prompt to Cowork ──
        log.info("[?] No specific command matched — sending as Cowork prompt")
        return self.prompt_cowork(text)

    # ─── Handler Methods ─────────────────────────────────────────────────

    def open_cowork(self, *args):
        """Open Claude Cowork in browser."""
        log.info("[WEB] Opening Claude Cowork...")
        return self._run_cc("chain",
            '"launch https://claude.ai; wait 3; screenshot"')

    def open_terminal(self, *args):
        """Open Windows Terminal."""
        log.info("[SYS] Opening Terminal...")
        return self._run_cc("chain",
            '"launch wt.exe; wait 2; screenshot"')

    def take_screenshot(self, *args):
        """Take a screenshot."""
        log.info("[CAM] Taking screenshot...")
        return self._run_cc("screenshot")

    def open_cowork_conversation(self, convo_name: str):
        """Open a specific Cowork conversation by name."""
        log.info(f"[NAV] Opening Cowork conversation: {convo_name}")
        # Focus browser, go to claude.ai, search for conversation
        return self._run_cc("chain",
            f'"launch https://claude.ai; wait 3; '
            f'click_text Search --window Chrome; wait 0.5; '
            f'type {convo_name}; wait 1; screenshot"')

    def prompt_cowork(self, prompt_text: str):
        """Type a prompt into Claude Cowork."""
        log.info(f"[>>] Prompting Cowork: {prompt_text[:80]}...")
        # Focus Cowork window, find the input area, type the prompt, send
        safe_text = prompt_text.replace('"', '\\"')
        return self._run_cc("chain",
            f'"focus Claude; wait 0.5; '
            f'click 960 900; wait 0.3; '
            f'type {safe_text}; wait 0.3; '
            f'key enter; wait 1; screenshot"')

    def prompt_claude_code(self, prompt_text: str):
        """Send a command to Claude Code in the terminal."""
        log.info(f"[>>] Prompting Claude Code: {prompt_text[:80]}...")
        terminal_title = CONFIG.get("claude_code_terminal", "Windows Terminal")
        safe_text = prompt_text.replace('"', '\\"')
        return self._run_cc("chain",
            f'"focus {terminal_title}; wait 0.5; '
            f'type {safe_text}; wait 0.3; '
            f'key enter; wait 1; screenshot"')

    def type_text(self, content: str):
        """Type text at current cursor position (voice-to-type)."""
        log.info(f"[KEY] Typing: {content[:60]}...")
        safe_text = content.replace('"', '\\"')
        return self._run_cc("type", f'"{safe_text}"')

    def search_web(self, query: str):
        """Open browser and search for something."""
        log.info(f"[WEB] Searching: {query}")
        safe_query = query.replace('"', '\\"')
        return self._run_cc("chain",
            f'"focus Chrome; wait 0.5; key ctrl+l; wait 0.2; '
            f'type https://www.google.com/search?q={safe_query}; '
            f'key enter; wait 3; screenshot"')

    def focus_window(self, window_name: str):
        """Focus/bring a window to foreground."""
        log.info(f"[WIN] Focusing: {window_name}")
        return self._run_cc("focus", f'"{window_name}"')

    # ─── Custom Command Management ──────────────────────────────────────

    def _handle_add_command(self, text: str):
        """Parse 'add command [name] triggers [t1,t2] action [action]'."""
        log.info(f"[+] Adding custom command from voice: {text}")
        # Simple parsing - for complex commands, use the command editor
        return {
            "status": "info",
            "message": "To add custom commands, edit commands/custom_commands.json "
                       "or use the Jarvis Command Manager (jarvis_manager.py)"
        }

    def _list_commands(self):
        """List all available commands."""
        built_in = [
            "open cowork", "open terminal", "screenshot",
            "open cowork convo [name]", "prompt cowork [text]",
            "prompt claude code [text]", "type [text]",
            "search [query]", "focus [window]",
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

                # Feed to OpenWakeWord
                prediction = self.oww_model.predict(audio_array)

                # Check if "hey_jarvis" score exceeds threshold
                scores = self.oww_model.get_keyword_predictions()
                jarvis_score = scores.get("hey_jarvis", 0)

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
