"""
Enhanced Jarvis Command Manager - GUI with Voice Integration
================================================================
Modern, robust GUI with voice command creation, testing, and management.
Features:
- Voice command creation (speech-to-text)
- Command testing with live feedback
- Token usage tracking
- Dark theme with modern styling
- Import/Export functionality
- Real-time Jarvis status
- Keyboard shortcuts
- Command templates
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
import threading
import queue
import time
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import wave
import pyaudio
import struct

# Import Jarvis components for voice integration
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from jarvis_core import WhisperTranscriber, HAS_OPENAI, whisper_client
    from jarvis_core import TOKEN_STATS, log_tokens, count_tokens
except ImportError:
    print("[!] Warning: Jarvis components not available")
    HAS_OPENAI = False
    whisper_client = None

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
COMMANDS_FILE = BASE_DIR / "commands" / "custom_commands.json"
CONFIG_FILE = BASE_DIR / "config.json"

# ─── Enhanced Colors (modern dark theme) ─────────────────────────────────────
C = {
    "bg": "#0a0e17",
    "bg2": "#111827", 
    "bg3": "#1a2235",
    "bg4": "#1e293b",
    "accent": "#00d4ff",
    "accent2": "#7c3aed",
    "accent3": "#10b981",
    "accent4": "#f59e0b",
    "text": "#e2e8f0",
    "dim": "#94a3b8",
    "border": "#334155",
    "red": "#ef4444",
    "yellow": "#fbbf24",
    "green": "#10b981",
    "entry_bg": "#1e293b",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "error": "#ef4444",
}

# ─── Voice Recording ─────────────────────────────────────────────────────────
class VoiceRecorder:
    def __init__(self, on_complete_callback):
        self.on_complete = on_complete_callback
        self.recording = False
        self.frames = []
        self.sample_rate = 16000
        self.chunk_size = 512
        self.energy_threshold = 300
        
    def start_recording(self):
        """Start recording audio."""
        self.frames = []
        self.recording = True
        
        def record():
            try:
                pa = pyaudio.PyAudio()
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                
                silence_start = None
                max_duration = 10  # 10 seconds max
                
                while self.recording and len(self.frames) < (self.sample_rate * max_duration // self.chunk_size):
                    try:
                        data = stream.read(self.chunk_size, exception_on_overflow=False)
                        self.frames.append(data)
                        
                        # Calculate RMS energy
                        samples = struct.unpack(f"{self.chunk_size}h", data)
                        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
                        
                        if rms < self.energy_threshold:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > 2.0:  # 2 seconds of silence
                                break
                        else:
                            silence_start = None
                            
                    except Exception as e:
                        print(f"[!] Audio recording error: {e}")
                        break
                
                stream.stop_stream()
                stream.close()
                pa.terminate()
                
                # Save to temp file
                if self.frames:
                    temp_path = tempfile.mktemp(suffix=".wav")
                    with wave.open(temp_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(b''.join(self.frames))
                    
                    self.on_complete(temp_path)
                    
            except Exception as e:
                print(f"[!] Voice recording error: {e}")
                self.on_complete(None)
        
        threading.Thread(target=record, daemon=True).start()
    
    def stop_recording(self):
        """Stop recording audio."""
        self.recording = False

# ─── Data Management ─────────────────────────────────────────────────────────
def load_commands():
    if COMMANDS_FILE.exists():
        with open(COMMANDS_FILE, "r") as f:
            return json.load(f)
    return {"commands": [], "aliases": {}}

def save_commands(data):
    COMMANDS_FILE.parent.mkdir(exist_ok=True)
    with open(COMMANDS_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Enhanced action templates
COMMAND_TEMPLATES = [
    {
        "name": "Open Website",
        "triggers": ["open website", "visit site"],
        "action": "chain: launch https://example.com",
        "description": "Opens a website in the default browser"
    },
    {
        "name": "Launch Application", 
        "triggers": ["launch app", "open program"],
        "action": "chain: launch notepad",
        "description": "Launches a desktop application"
    },
    {
        "name": "Search Google",
        "triggers": ["search", "google search"],
        "action": "chain: launch https://www.google.com/search?q=",
        "description": "Searches Google for a query"
    },
    {
        "name": "Take Screenshot",
        "triggers": ["screenshot", "capture screen"],
        "action": "screenshot",
        "description": "Takes a screenshot of the current screen"
    },
    {
        "name": "Type Text",
        "triggers": ["type", "write text"],
        "action": "chain: type ",
        "description": "Types text at the current cursor position"
    }
]

# Available built-in actions
ACTIONS = [
    ("open_cowork", "Open Claude Cowork in browser"),
    ("open_terminal", "Open Windows Terminal"),
    ("screenshot", "Take a screenshot"),
    ("open_cowork_convo", "Open a Cowork conversation by name"),
    ("prompt_cowork", "Send a prompt to Cowork"),
    ("prompt_claude_code", "Send a command to Claude Code"),
    ("type_text", "Type text at cursor position"),
    ("search_web", "Search Google"),
    ("focus_window", "Focus/bring a window to front"),
]

# ─── Enhanced Main App ─────────────────────────────────────────────────────────
class EnhancedJarvisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("JARVIS Command Manager - Enhanced")
        self.geometry("1200x800")
        self.configure(bg=C["bg"])
        self.resizable(True, True)
        self.minsize(900, 600)
        
        # Data
        self.data = load_commands()
        self.selected_index = None
        self.voice_recorder = None
        self.transcriber = WhisperTranscriber() if HAS_OPENAI else None
        
        # Status tracking
        self.status_queue = queue.Queue()
        self.jarvis_status = "Offline"
        self.last_command_result = None
        
        # Setup
        self._setup_styles()
        self._build_ui()
        self._refresh_list()
        self._setup_keyboard_shortcuts()
        self._start_status_updater()
        
        # Check Jarvis status
        self._check_jarvis_status()

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Enhanced styles
        style.configure("Dark.TFrame", background=C["bg"])
        style.configure("Card.TFrame", background=C["bg2"])
        style.configure("Accent.TFrame", background=C["accent3"])
        style.configure("Dark.TLabel", background=C["bg"], foreground=C["text"],
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=C["bg"], foreground=C["accent"],
                        font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background=C["bg"], foreground=C["dim"],
                        font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background=C["bg2"], foreground=C["text"],
                        font=("Segoe UI", 12, "bold"))
        style.configure("CardDim.TLabel", background=C["bg2"], foreground=C["dim"],
                        font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=C["bg4"], foreground=C["text"],
                        font=("Segoe UI", 9))
        style.configure("Accent.TButton", background=C["accent"], foreground="#000",
                        font=("Segoe UI", 10, "bold"), padding=(16, 8))
        style.map("Accent.TButton",
                  background=[("active", C["accent2"]), ("pressed", C["accent2"])])
        style.configure("Success.TButton", background=C["success"], foreground="#fff",
                        font=("Segoe UI", 10, "bold"), padding=(16, 8))
        style.configure("Danger.TButton", background=C["red"], foreground="#fff",
                        font=("Segoe UI", 10), padding=(12, 6))
        style.map("Danger.TButton", background=[("active", "#dc2626")])
        style.configure("Ghost.TButton", background=C["bg3"], foreground=C["text"],
                        font=("Segoe UI", 10), padding=(12, 6))
        style.map("Ghost.TButton", background=[("active", C["border"])])
        style.configure("Voice.TButton", background=C["accent2"], foreground="#fff",
                        font=("Segoe UI", 10, "bold"), padding=(16, 8))

    def _build_ui(self):
        # ── Header with Status ──
        header = ttk.Frame(self, style="Dark.TFrame")
        header.pack(fill="x", padx=24, pady=(20, 0))

        left_header = ttk.Frame(header, style="Dark.TFrame")
        left_header.pack(side="left")
        
        ttk.Label(left_header, text="JARVIS", style="Title.TLabel").pack(side="left")
        ttk.Label(left_header, text="  Enhanced Command Manager", style="Subtitle.TLabel").pack(side="left", pady=(6, 0))

        # Status indicator
        status_frame = ttk.Frame(header, style="Dark.TFrame")
        status_frame.pack(side="right", padx=(20, 0))
        
        self.status_label = ttk.Label(status_frame, text="● Offline", 
                                   style="Status.TLabel", foreground=C["red"])
        self.status_label.pack(side="left", padx=(0, 8))
        
        # Token usage
        self.token_label = ttk.Label(status_frame, text="Tokens: 0", style="Status.TLabel")
        self.token_label.pack(side="left")

        # ── Toolbar ──
        toolbar = ttk.Frame(self, style="Dark.TFrame")
        toolbar.pack(fill="x", padx=24, pady=(16, 0))

        # Left side buttons
        left_toolbar = ttk.Frame(toolbar, style="Dark.TFrame")
        left_toolbar.pack(side="left")
        
        ttk.Button(left_toolbar, text="+ New Command", style="Accent.TButton",
                   command=self._new_command).pack(side="left", padx=(0, 8))
        ttk.Button(left_toolbar, text="+ Voice Command", style="Voice.TButton",
                   command=self._voice_command).pack(side="left", padx=(0, 8))
        ttk.Button(left_toolbar, text="+ New Alias", style="Ghost.TButton",
                   command=self._new_alias).pack(side="left", padx=(0, 8))
        
        # Right side buttons
        right_toolbar = ttk.Frame(toolbar, style="Dark.TFrame")
        right_toolbar.pack(side="right")
        
        ttk.Button(right_toolbar, text="Import", style="Ghost.TButton",
                   command=self._import_commands).pack(side="left", padx=(0, 8))
        ttk.Button(right_toolbar, text="Export", style="Ghost.TButton",
                   command=self._export_commands).pack(side="left", padx=(0, 8))
        ttk.Button(right_toolbar, text="Templates", style="Ghost.TButton",
                   command=self._show_templates).pack(side="left")

        # ── Search and Filter ──
        search_frame = ttk.Frame(self, style="Dark.TFrame")
        search_frame.pack(fill="x", padx=24, pady=(16, 0))

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *a: self._refresh_list())
        search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                                bg=C["entry_bg"], fg=C["text"], insertbackground=C["accent"],
                                font=("Segoe UI", 11), bd=0, highlightthickness=1,
                                highlightcolor=C["accent"], highlightbackground=C["border"])
        search_entry.pack(fill="x", ipady=8, ipadx=12)
        search_entry.insert(0, "")
        search_entry.bind("<FocusIn>", lambda e: None)

        # ── Main Content Area ──
        content_frame = ttk.Frame(self, style="Dark.TFrame")
        content_frame.pack(fill="both", expand=True, padx=24, pady=(16, 0))

        # Command list (left side)
        list_container = ttk.Frame(content_frame, style="Dark.TFrame")
        list_container.pack(side="left", fill="both", expand=True)

        ttk.Label(list_container, text="Commands", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))
        
        # Scrollable command list
        list_frame = ttk.Frame(list_container, style="Dark.TFrame")
        list_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(list_frame, bg=C["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        self.scroll_inner = ttk.Frame(canvas, style="Dark.TFrame")

        self.scroll_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.scroll_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas = canvas

        # Details panel (right side)
        details_panel = ttk.Frame(content_frame, style="Card.TFrame", width=300)
        details_panel.pack(side="right", fill="y", padx=(16, 0))
        details_panel.pack_propagate(False)

        ttk.Label(details_panel, text="Details", style="CardTitle.TLabel").pack(anchor="w", padx=16, pady=(16, 12))
        
        self.details_text = tk.Text(details_panel, bg=C["bg3"], fg=C["text"], 
                                   font=("Segoe UI", 10), bd=0, highlightthickness=0,
                                   wrap="word", height=15, width=35)
        self.details_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        
        # Test button
        self.test_btn = ttk.Button(details_panel, text="Test Command", style="Success.TButton",
                                  command=self._test_command, state="disabled")
        self.test_btn.pack(padx=16, pady=(0, 16))

        # ── Aliases section ──
        alias_container = ttk.Frame(self, style="Dark.TFrame")
        alias_container.pack(fill="x", padx=24, pady=(12, 0))

        alias_header = ttk.Frame(alias_container, style="Dark.TFrame")
        alias_header.pack(fill="x")
        
        ttk.Label(alias_header, text="Aliases", style="Dark.TLabel",
                  font=("Segoe UI", 12, "bold")).pack(side="left")
        
        self.alias_count = ttk.Label(alias_header, text="(0)", style="Subtitle.TLabel")
        self.alias_count.pack(side="left", padx=(8, 0))

        self.alias_frame = ttk.Frame(alias_container, style="Dark.TFrame")
        self.alias_frame.pack(fill="x", pady=(8, 0))

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Ready")
        status = tk.Label(self, textvariable=self.status_var, bg=C["bg4"],
                         fg=C["dim"], font=("Segoe UI", 9), anchor="w", padx=12, pady=6)
        status.pack(fill="x", side="bottom")

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        self.bind("<Control-n>", lambda e: self._new_command())
        self.bind("<Control-v>", lambda e: self._voice_command())
        self.bind("<Control-i>", lambda e: self._import_commands())
        self.bind("<Control-e>", lambda e: self._export_commands())
        self.bind("<Delete>", lambda e: self._delete_selected())
        self.bind("<F5>", lambda e: self._refresh_list())

    def _start_status_updater(self):
        """Start background status updater."""
        def update_status():
            while True:
                try:
                    # Update token display
                    total_input = TOKEN_STATS["openai_chat"]["input_tokens"] + TOKEN_STATS["claude"]["input_tokens"]
                    total_output = TOKEN_STATS["openai_chat"]["output_tokens"] + TOKEN_STATS["claude"]["output_tokens"]
                    total_tokens = total_input + total_tokens
                    
                    self.token_label.config(text=f"Tokens: {total_tokens:,}")
                    
                    # Check Jarvis status periodically
                    if int(time.time()) % 10 == 0:  # Every 10 seconds
                        self._check_jarvis_status()
                    
                    time.sleep(1)
                except:
                    pass
        
        threading.Thread(target=update_status, daemon=True).start()

    def _check_jarvis_status(self):
        """Check if Jarvis is running."""
        try:
            # Simple check - could be enhanced with actual IPC
            result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"], 
                                  capture_output=True, text=True, timeout=5)
            if "python.exe" in result.stdout:
                self.jarvis_status = "Online"
                self.status_label.config(text="● Online", foreground=C["success"])
            else:
                self.jarvis_status = "Offline"
                self.status_label.config(text="● Offline", foreground=C["red"])
        except:
            self.jarvis_status = "Unknown"
            self.status_label.config(text="● Unknown", foreground=C["yellow"])

    def _refresh_list(self):
        """Rebuild the command card list."""
        # Clear existing
        for widget in self.scroll_inner.winfo_children():
            widget.destroy()
        for widget in self.alias_frame.winfo_children():
            widget.destroy()

        search = self.search_var.get().lower()
        commands = self.data.get("commands", [])

        if not commands:
            lbl = tk.Label(self.scroll_inner, 
                          text="No custom commands yet.\\n\\n• Click '+ New Command' to create one\\n• Click '+ Voice Command' to create with speech\\n• Try 'Templates' for examples",
                          bg=C["bg"], fg=C["dim"], font=("Segoe UI", 11), justify="center")
            lbl.pack(pady=40)
        else:
            for i, cmd in enumerate(commands):
                # Filter by search
                searchable = f"{cmd.get('name','')} {' '.join(cmd.get('triggers',[]))} {cmd.get('action','')} {cmd.get('description','')}".lower()
                if search and search not in searchable:
                    continue
                self._build_card(i, cmd)

        # Aliases
        aliases = self.data.get("aliases", {})
        if aliases:
            for alias, target in aliases.items():
                row = tk.Frame(self.alias_frame, bg=C["bg3"], padx=12, pady=6)
                row.pack(fill="x", pady=2)
                tk.Label(row, text=f'"{alias}"', bg=C["bg3"], fg=C["accent"],
                        font=("Consolas", 10, "bold")).pack(side="left")
                tk.Label(row, text=f"  →  {target}", bg=C["bg3"], fg=C["dim"],
                        font=("Segoe UI", 10)).pack(side="left")
                del_btn = tk.Button(row, text="×", bg=C["bg3"], fg=C["red"],
                                   font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2",
                                   command=lambda a=alias: self._delete_alias(a))
                del_btn.pack(side="right")

        # Update counts
        count = len(commands)
        alias_count = len(aliases)
        self.status_var.set(f"{count} command{'s' if count != 1 else ''}  |  {alias_count} alias{'es' if alias_count != 1 else ''}  |  {self.jarvis_status}")
        self.alias_count.config(text=f"({alias_count})")

    def _build_card(self, index, cmd):
        """Build an enhanced command card."""
        card = tk.Frame(self.scroll_inner, bg=C["bg2"], padx=16, pady=14,
                       highlightbackground=C["border"], highlightthickness=1)
        card.pack(fill="x", pady=4)
        
        # Bind click to select
        def select_card(e):
            self._select_command(index)
        card.bind("<Button-1>", select_card)
        
        # Add selection indicator
        self.selection_indicator = None

        # Top row: name + action badge
        top = tk.Frame(card, bg=C["bg2"])
        top.pack(fill="x")

        tk.Label(top, text=cmd.get("name", "Untitled"), bg=C["bg2"], fg=C["text"],
                font=("Segoe UI", 12, "bold")).pack(side="left")

        action = cmd.get("action", "")
        if action.startswith("chain:"):
            badge_color = C["accent4"]
            badge_text = "Chain Command"
        elif action in [a[0] for a in ACTIONS]:
            badge_color = C["accent3"]
            badge_text = "Built-in"
        else:
            badge_color = C["accent"]
            badge_text = "Custom"
            
        badge = tk.Label(top, text=badge_text, bg=C["bg2"], fg=badge_color,
                        font=("Segoe UI", 8, "bold"))
        badge.pack(side="right")

        # Triggers
        triggers = cmd.get("triggers", [])
        trigger_text = "  |  ".join([f'"{t}"' for t in triggers])
        tk.Label(card, text=trigger_text, bg=C["bg2"], fg=C["accent"],
                font=("Consolas", 10)).pack(anchor="w", pady=(6, 0))

        # Description
        desc = cmd.get("description", "")
        if desc:
            tk.Label(card, text=desc, bg=C["bg2"], fg=C["dim"],
                    font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))

        # Buttons
        btn_row = tk.Frame(card, bg=C["bg2"])
        btn_row.pack(fill="x", pady=(10, 0))

        tk.Button(btn_row, text="Edit", bg=C["bg3"], fg=C["text"],
                            font=("Segoe UI", 9), bd=0, padx=12, pady=4, cursor="hand2",
                            command=lambda i=index: self._edit_command(i)).pack(side="left", padx=(0, 6))

        tk.Button(btn_row, text="Test", bg=C["success"], fg="#fff",
                           font=("Segoe UI", 9), bd=0, padx=12, pady=4, cursor="hand2",
                           command=lambda i=index: self._test_command_by_index(i)).pack(side="left", padx=(0, 6))

        tk.Button(btn_row, text="Duplicate", bg=C["bg3"], fg=C["text"],
                           font=("Segoe UI", 9), bd=0, padx=12, pady=4, cursor="hand2",
                           command=lambda i=index: self._duplicate_command(i)).pack(side="left", padx=(0, 6))

        tk.Button(btn_row, text="Delete", bg=C["bg3"], fg=C["red"],
                           font=("Segoe UI", 9), bd=0, padx=12, pady=4, cursor="hand2",
                           command=lambda i=index: self._delete_command(i)).pack(side="left")

    def _select_command(self, index):
        """Select a command and show details."""
        self.selected_index = index
        cmd = self.data["commands"][index]
        
        # Update details panel
        self.details_text.delete(1.0, tk.END)
        details = f"Name: {cmd.get('name', 'N/A')}\\n"
        details += f"Triggers: {', '.join(cmd.get('triggers', []))}\\n"
        details += f"Action: {cmd.get('action', 'N/A')}\\n"
        details += f"Description: {cmd.get('description', 'No description')}\\n\\n"
        
        # Add action explanation
        action = cmd.get('action', '')
        if action.startswith('chain:'):
            details += "This is a chain command that will execute multiple actions in sequence.\\n\\n"
            chain_actions = action[6:]  # Remove 'chain:' prefix
            details += f"Chain: {chain_actions}"
        elif action in [a[0] for a in ACTIONS]:
            for name, desc in ACTIONS:
                if action == name:
                    details += f"Built-in action: {desc}"
                    break
        else:
            details += "Custom action type"
        
        self.details_text.insert(1.0, details)
        self.test_btn.config(state="normal")

    def _voice_command(self):
        """Create command using voice input."""
        if not HAS_OPENAI or not whisper_client:
            messagebox.showwarning("Voice Not Available", 
                                  "Voice transcription requires OpenAI API key in config.json")
            return
            
        dialog = VoiceCommandDialog(self, self._on_voice_complete)
        self.wait_window(dialog)

    def _on_voice_complete(self, result):
        """Handle voice command creation result."""
        if result:
            # Parse the voice input to extract command details
            # This is a simple implementation - could be enhanced with NLP
            self.status_var.set(f"Voice command created: {result['name']}")
            self.data.setdefault("commands", []).append(result)
            save_commands(self.data)
            self._refresh_list()

    def _test_command(self):
        """Test the selected command."""
        if self.selected_index is not None:
            self._test_command_by_index(self.selected_index)

    def _test_command_by_index(self, index):
        """Test a specific command."""
        cmd = self.data["commands"][index]
        action = cmd.get("action", "")
        
        if action.startswith("chain:"):
            # Test chain command
            chain_cmd = action[6:]
            try:
                result = subprocess.run(
                    f'python "{CONFIG_FILE.parent.parent / "claudebridge" / "skills" / "cc.py"}" chain "{chain_cmd}"',
                    shell=True, capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    messagebox.showinfo("Test Success", f"Command executed successfully!\\n\\nOutput:\\n{result.stdout}")
                else:
                    messagebox.showwarning("Test Failed", f"Command failed!\\n\\nError:\\n{result.stderr}")
            except Exception as e:
                messagebox.showerror("Test Error", f"Error testing command: {e}")
        else:
            messagebox.showinfo("Test", f"Would execute: {action}")

    def _new_command(self):
        """Open dialog to create a new command."""
        dialog = EnhancedCommandDialog(self, title="New Command")
        self.wait_window(dialog)
        if dialog.result:
            self.data.setdefault("commands", []).append(dialog.result)
            save_commands(self.data)
            self._refresh_list()
            self.status_var.set(f"Created command: {dialog.result['name']}")

    def _edit_command(self, index):
        """Open dialog to edit an existing command."""
        cmd = self.data["commands"][index]
        dialog = EnhancedCommandDialog(self, title="Edit Command", existing=cmd)
        self.wait_window(dialog)
        if dialog.result:
            self.data["commands"][index] = dialog.result
            save_commands(self.data)
            self._refresh_list()
            self.status_var.set(f"Updated command: {dialog.result['name']}")

    def _duplicate_command(self, index):
        cmd = self.data["commands"][index].copy()
        cmd["name"] = cmd["name"] + " (copy)"
        self.data["commands"].append(cmd)
        save_commands(self.data)
        self._refresh_list()
        self.status_var.set(f"Duplicated: {cmd['name']}")

    def _delete_command(self, index):
        cmd = self.data["commands"][index]
        if messagebox.askyesno("Delete Command", f"Delete '{cmd['name']}'?"):
            self.data["commands"].pop(index)
            save_commands(self.data)
            self._refresh_list()
            self.status_var.set(f"Deleted: {cmd['name']}")

    def _delete_selected(self):
        """Delete selected command."""
        if self.selected_index is not None:
            self._delete_command(self.selected_index)

    def _new_alias(self):
        dialog = AliasDialog(self)
        self.wait_window(dialog)
        if dialog.result:
            alias, target = dialog.result
            self.data.setdefault("aliases", {})[alias] = target
            save_commands(self.data)
            self._refresh_list()
            self.status_var.set(f"Added alias: '{alias}' -> '{target}'")

    def _delete_alias(self, alias):
        if messagebox.askyesno("Delete Alias", f"Delete alias '{alias}'?"):
            self.data.get("aliases", {}).pop(alias, None)
            save_commands(self.data)
            self._refresh_list()
            self.status_var.set(f"Deleted alias: {alias}")

    def _show_templates(self):
        """Show command templates dialog."""
        dialog = TemplatesDialog(self, COMMAND_TEMPLATES)
        self.wait_window(dialog)
        if dialog.result:
            # Add selected template
            self.data.setdefault("commands", []).append(dialog.result)
            save_commands(self.data)
            self._refresh_list()
            self.status_var.set(f"Added template: {dialog.result['name']}")

    def _import_commands(self):
        """Import commands from JSON file."""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Import Commands",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    imported_data = json.load(f)
                
                # Merge with existing data
                if "commands" in imported_data:
                    self.data.setdefault("commands", []).extend(imported_data["commands"])
                if "aliases" in imported_data:
                    self.data.setdefault("aliases", {}).update(imported_data["aliases"])
                
                save_commands(self.data)
                self._refresh_list()
                self.status_var.set(f"Imported {len(imported_data.get('commands', []))} commands")
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import: {e}")

    def _export_commands(self):
        """Export commands to JSON file."""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            title="Export Commands",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.data, f, indent=4)
                self.status_var.set(f"Exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")


# ─── Enhanced Dialogs ─────────────────────────────────────────────────────────
class VoiceCommandDialog(tk.Toplevel):
    def __init__(self, parent, on_complete):
        super().__init__(parent)
        self.title("Voice Command Creation")
        self.geometry("500x400")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        self.result = None
        self.on_complete = on_complete
        self.recording = False
        self.voice_recorder = None
        
        self._build_ui()
        self._center_window()

    def _center_window(self):
        """Center the dialog on parent."""
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (self.winfo_width() // 2)
        y = (self.winfo_screenheight() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        # Title
        tk.Label(self, text="Create Command with Voice", bg=C["bg"], fg=C["text"],
                font=("Segoe UI", 16, "bold")).pack(pady=(20, 10))
        
        tk.Label(self, text="Say the command name and triggers naturally", bg=C["bg"], fg=C["dim"],
                font=("Segoe UI", 10)).pack(pady=(0, 20))

        # Recording area
        self.record_frame = tk.Frame(self, bg=C["bg2"], padx=20, pady=20)
        self.record_frame.pack(fill="x", padx=20)

        self.record_btn = tk.Button(self.record_frame, text="🎤 Start Recording", 
                                   bg=C["accent2"], fg="#fff", font=("Segoe UI", 12, "bold"),
                                   bd=0, padx=20, pady=15, cursor="hand2",
                                   command=self._toggle_recording)
        self.record_btn.pack()

        self.status_label = tk.Label(self.record_frame, text="Click to start recording", 
                                    bg=C["bg2"], fg=C["dim"], font=("Segoe UI", 10))
        self.status_label.pack(pady=(10, 0))

        # Transcription result
        self.trans_frame = tk.Frame(self, bg=C["bg"])
        self.trans_frame.pack(fill="both", expand=True, padx=20, pady=20)

        tk.Label(self.trans_frame, text="Transcription:", bg=C["bg"], fg=C["dim"],
                font=("Segoe UI", 10)).pack(anchor="w")

        self.trans_text = tk.Text(self.trans_frame, bg=C["entry_bg"], fg=C["text"],
                                  font=("Segoe UI", 10), bd=0, highlightthickness=1,
                                  highlightcolor=C["accent"], height=8)
        self.trans_text.pack(fill="both", expand=True, pady=(5, 0))

        # Buttons
        btn_frame = tk.Frame(self, bg=C["bg"])
        btn_frame.pack(fill="x", padx=20, pady=(0, 20))

        tk.Button(btn_frame, text="Cancel", bg=C["bg3"], fg=C["text"],
                 font=("Segoe UI", 10), bd=0, padx=16, pady=8, cursor="hand2",
                 command=self.destroy).pack(side="right")
        tk.Button(btn_frame, text="Create Command", bg=C["accent"], fg="#000",
                 font=("Segoe UI", 10, "bold"), bd=0, padx=20, pady=8, cursor="hand2",
                 command=self._create_command, state="disabled").pack(side="right", padx=(0, 8))

    def _toggle_recording(self):
        """Toggle recording state."""
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        """Start voice recording."""
        self.recording = True
        self.record_btn.config(text="⏹️ Stop Recording", bg=C["red"])
        self.status_label.config(text="Recording... Speak clearly")
        
        def on_complete(audio_path):
            if audio_path:
                self.status_label.config(text="Transcribing...")
                self._transcribe_audio(audio_path)
            else:
                self.status_label.config(text="Recording failed")
                self.recording = False
                self.record_btn.config(text="🎤 Start Recording", bg=C["accent2"])
        
        self.voice_recorder = VoiceRecorder(on_complete)
        self.voice_recorder.start_recording()

    def _stop_recording(self):
        """Stop voice recording."""
        self.recording = False
        if self.voice_recorder:
            self.voice_recorder.stop_recording()
        self.record_btn.config(text="🎤 Start Recording", bg=C["accent2"])
        self.status_label.config(text="Processing...")

    def _transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper."""
        try:
            transcriber = WhisperTranscriber()
            text = transcriber.transcribe(Path(audio_path))
            
            # Clean up temp file
            try:
                os.unlink(audio_path)
            except:
                pass
            
            if text:
                self.trans_text.delete(1.0, tk.END)
                self.trans_text.insert(1.0, text)
                self.status_label.config(text="Transcription complete!")
                
                # Enable create button
                for widget in self.winfo_children():
                    if isinstance(widget, tk.Frame):
                        for child in widget.winfo_children():
                            if isinstance(child, tk.Button) and "Create Command" in child.cget("text"):
                                child.config(state="normal")
            else:
                self.status_label.config(text="No speech detected")
                
        except Exception as e:
            self.status_label.config(text=f"Transcription error: {e}")

    def _create_command(self):
        """Create command from transcription."""
        text = self.trans_text.get(1.0, tk.END).strip()
        if not text:
            return
        
        # Simple parsing - could be enhanced with NLP
        # Assume format: "Create command called [name] with triggers [triggers] that does [description]"
        words = text.lower().split()
        
        # Extract name (after "called" or "named")
        name = "Voice Command"
        triggers = []
        description = text
        
        # Simple heuristic extraction
        if "called" in words or "named" in words:
            try:
                idx = max(words.index("called") if "called" in words else -1,
                         words.index("named") if "named" in words else -1)
                if idx + 1 < len(words):
                    # Get next few words as name
                    name_words = []
                    for i in range(idx + 1, min(idx + 4, len(words))):
                        if words[i] not in ["with", "that", "and", "to"]:
                            name_words.append(words[i])
                        else:
                            break
                    name = " ".join(name_words).title()
            except:
                pass
        
        # Extract triggers (after "triggers" or "when I say")
        if "triggers" in words or "when i say" in text.lower():
            # Simple extraction - could be improved
            triggers = [name.lower()]
        
        self.result = {
            "name": name,
            "triggers": triggers or [name.lower()],
            "action": "chain: type " + text,  # Default action
            "description": description[:100] + "..." if len(description) > 100 else description
        }
        
        self.on_complete(self.result)
        self.destroy()


class EnhancedCommandDialog(tk.Toplevel):
    def __init__(self, parent, title="Command", existing=None):
        super().__init__(parent)
        self.title(title)
        self.geometry("600x600")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        self.result = None
        self.transient(parent)
        self.grab_set()

        self._build_ui(existing)
        self._center_window()

    def _center_window(self):
        """Center the dialog on parent."""
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (self.winfo_width() // 2)
        y = (self.winfo_screenheight() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _build_ui(self, existing):
        pad = {"padx": 20, "pady": (0, 0)}

        # Name
        tk.Label(self, text="Command Name", bg=C["bg"], fg=C["dim"],
                font=("Segoe UI", 9)).pack(anchor="w", **pad, pady=(20, 4))
        self.name_var = tk.StringVar(value=existing["name"] if existing else "")
        tk.Entry(self, textvariable=self.name_var, bg=C["entry_bg"], fg=C["text"],
                insertbackground=C["accent"], font=("Segoe UI", 11), bd=0,
                highlightthickness=1, highlightcolor=C["accent"],
                highlightbackground=C["border"]).pack(fill="x", padx=20, ipady=8)

        # Triggers
        tk.Label(self, text="Trigger Phrases (comma-separated)", bg=C["bg"], fg=C["dim"],
                font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(16, 4))
        self.triggers_var = tk.StringVar(
            value=", ".join(existing["triggers"]) if existing else "")
        tk.Entry(self, textvariable=self.triggers_var, bg=C["entry_bg"], fg=C["accent"],
                insertbackground=C["accent"], font=("Consolas", 10), bd=0,
                highlightthickness=1, highlightcolor=C["accent"],
                highlightbackground=C["border"]).pack(fill="x", padx=20, ipady=8)

        # Action type
        tk.Label(self, text="Action", bg=C["bg"], fg=C["dim"],
                font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(16, 4))

        action_frame = tk.Frame(self, bg=C["bg"])
        action_frame.pack(fill="x", padx=20)

        self.action_var = tk.StringVar(value=existing.get("action", "") if existing else "")
        action_combo = ttk.Combobox(action_frame, textvariable=self.action_var,
                                    values=[a[0] for a in ACTIONS] + ["chain:"],
                                    font=("Consolas", 10), state="normal")
        action_combo.pack(fill="x", ipady=4)

        # Action help text
        self.help_var = tk.StringVar(value="Select an action or type a custom chain:... command")
        tk.Label(self, textvariable=self.help_var, bg=C["bg"], fg=C["dim"],
                font=("Segoe UI", 8)).pack(anchor="w", padx=20, pady=(4, 0))

        def _update_help(*a):
            action = self.action_var.get()
            for name, desc in ACTIONS:
                if action == name:
                    self.help_var.set(desc)
                    return
            if action.startswith("chain:"):
                self.help_var.set("Custom cc.py chain — semicolon-separated actions")
            else:
                self.help_var.set("")
        self.action_var.trace_add("write", _update_help)

        # Description
        tk.Label(self, text="Description (optional)", bg=C["bg"], fg=C["dim"],
                font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(16, 4))
        self.desc_var = tk.StringVar(value=existing.get("description", "") if existing else "")
        tk.Entry(self, textvariable=self.desc_var, bg=C["entry_bg"], fg=C["text"],
                insertbackground=C["accent"], font=("Segoe UI", 10), bd=0,
                highlightthickness=1, highlightcolor=C["accent"],
                highlightbackground=C["border"]).pack(fill="x", padx=20, ipady=8)

        # Category
        tk.Label(self, text="Category (optional)", bg=C["bg"], fg=C["dim"],
                font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(16, 4))
        self.category_var = tk.StringVar(value=existing.get("category", "") if existing else "")
        category_combo = ttk.Combobox(self, textvariable=self.category_var,
                                      values=["Productivity", "Entertainment", "System", "Web", "Custom"],
                                      font=("Segoe UI", 10), state="normal")
        category_combo.pack(fill="x", padx=20, ipady=4)

        # Buttons
        btn_frame = tk.Frame(self, bg=C["bg"])
        btn_frame.pack(fill="x", padx=20, pady=(24, 20))

        tk.Button(btn_frame, text="Cancel", bg=C["bg3"], fg=C["text"],
                 font=("Segoe UI", 10), bd=0, padx=16, pady=8, cursor="hand2",
                 command=self.destroy).pack(side="right")
        tk.Button(btn_frame, text="Save Command", bg=C["accent"], fg="#000",
                 font=("Segoe UI", 10, "bold"), bd=0, padx=20, pady=8, cursor="hand2",
                 command=self._save).pack(side="right", padx=(0, 8))

    def _save(self):
        name = self.name_var.get().strip()
        triggers = [t.strip() for t in self.triggers_var.get().split(",") if t.strip()]
        action = self.action_var.get().strip()

        if not name:
            messagebox.showwarning("Missing", "Command name is required")
            return
        if not triggers:
            messagebox.showwarning("Missing", "At least one trigger phrase is required")
            return
        if not action:
            messagebox.showwarning("Missing", "Action is required")
            return

        self.result = {
            "name": name,
            "triggers": triggers,
            "action": action,
            "description": self.desc_var.get().strip(),
            "category": self.category_var.get().strip()
        }
        self.destroy()


class AliasDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("New Alias")
        self.geometry("420x250")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        self.result = None
        self.transient(parent)
        self.grab_set()

        tk.Label(self, text="Short Name (what you say)", bg=C["bg"], fg=C["dim"],
                font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(20, 4))
        self.alias_var = tk.StringVar()
        tk.Entry(self, textvariable=self.alias_var, bg=C["entry_bg"], fg=C["accent"],
                insertbackground=C["accent"], font=("Consolas", 11), bd=0,
                highlightthickness=1, highlightcolor=C["accent"],
                highlightbackground=C["border"]).pack(fill="x", padx=20, ipady=8)

        tk.Label(self, text="Maps To (full name)", bg=C["bg"], fg=C["dim"],
                font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(16, 4))
        self.target_var = tk.StringVar()
        tk.Entry(self, textvariable=self.target_var, bg=C["entry_bg"], fg=C["text"],
                insertbackground=C["accent"], font=("Segoe UI", 11), bd=0,
                highlightthickness=1, highlightcolor=C["accent"],
                highlightbackground=C["border"]).pack(fill="x", padx=20, ipady=8)

        btn_frame = tk.Frame(self, bg=C["bg"])
        btn_frame.pack(fill="x", padx=20, pady=(20, 20))
        tk.Button(btn_frame, text="Cancel", bg=C["bg3"], fg=C["text"],
                 font=("Segoe UI", 10), bd=0, padx=16, pady=8,
                 command=self.destroy).pack(side="right")
        tk.Button(btn_frame, text="Save Alias", bg=C["accent"], fg="#000",
                 font=("Segoe UI", 10, "bold"), bd=0, padx=20, pady=8,
                 command=self._save).pack(side="right", padx=(0, 8))

    def _save(self):
        alias = self.alias_var.get().strip().lower()
        target = self.target_var.get().strip()
        if not alias or not target:
            messagebox.showwarning("Missing", "Both fields are required")
            return
        self.result = (alias, target)
        self.destroy()


class TemplatesDialog(tk.Toplevel):
    def __init__(self, parent, templates):
        super().__init__(parent)
        self.title("Command Templates")
        self.geometry("700x500")
        self.configure(bg=C["bg"])
        self.result = None
        self.templates = templates
        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self._center_window()

    def _center_window(self):
        """Center the dialog on parent."""
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (self.winfo_width() // 2)
        y = (self.winfo_screenheight() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        # Title
        tk.Label(self, text="Command Templates", bg=C["bg"], fg=C["text"],
                font=("Segoe UI", 16, "bold")).pack(pady=(20, 10))

        # Template list
        list_frame = tk.Frame(self, bg=C["bg"])
        list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        canvas = tk.Canvas(list_frame, bg=C["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="Dark.TFrame")

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add templates
        for i, template in enumerate(self.templates):
            card = tk.Frame(inner, bg=C["bg2"], padx=16, pady=12,
                           highlightbackground=C["border"], highlightthickness=1)
            card.pack(fill="x", pady=4, padx=10)

            # Template name
            tk.Label(card, text=template["name"], bg=C["bg2"], fg=C["text"],
                    font=("Segoe UI", 12, "bold")).pack(anchor="w")

            # Triggers
            triggers = ", ".join(template["triggers"])
            tk.Label(card, text=f"Triggers: {triggers}", bg=C["bg2"], fg=C["accent"],
                    font=("Consolas", 9)).pack(anchor="w", pady=(4, 0))

            # Description
            tk.Label(card, text=template["description"], bg=C["bg2"], fg=C["dim"],
                    font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))

            # Action
            tk.Label(card, text=f"Action: {template['action']}", bg=C["bg2"], fg=C["accent3"],
                    font=("Consolas", 8)).pack(anchor="w", pady=(4, 0))

            # Use button
            tk.Button(card, text="Use Template", bg=C["accent"], fg="#000",
                     font=("Segoe UI", 9, "bold"), bd=0, padx=12, pady=6, cursor="hand2",
                     command=lambda t=template: self._use_template(t)).pack(anchor="e", pady=(8, 0))

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Close button
        tk.Button(self, text="Close", bg=C["bg3"], fg=C["text"],
                 font=("Segoe UI", 10), bd=0, padx=20, pady=8, cursor="hand2",
                 command=self.destroy).pack(pady=(0, 20))

    def _use_template(self, template):
        """Use a template."""
        self.result = template.copy()
        self.destroy()


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = EnhancedJarvisGUI()
    app.mainloop()
