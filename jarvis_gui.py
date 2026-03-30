"""
Jarvis Command Manager — GUI Edition
======================================
Beautiful dark-themed GUI for managing voice commands.
Launch via: python jarvis_gui.py
Or say: "Hey Jarvis, open command manager"
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
from pathlib import Path
import sys
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
COMMANDS_FILE = BASE_DIR / "commands" / "custom_commands.json"
CONFIG_FILE = BASE_DIR / "config.json"

# ─── Colors (dark theme) ─────────────────────────────────────────────────────
C = {
    "bg":       "#0a0e17",
    "bg2":      "#111827",
    "bg3":      "#1a2235",
    "accent":   "#00d4ff",
    "accent2":  "#7c3aed",
    "accent3":  "#10b981",
    "text":     "#e2e8f0",
    "dim":      "#94a3b8",
    "border":   "#1e293b",
    "red":      "#ef4444",
    "yellow":   "#fbbf24",
    "entry_bg": "#1e293b",
}

# ─── Data ─────────────────────────────────────────────────────────────────────
def load_commands():
    if COMMANDS_FILE.exists():
        with open(COMMANDS_FILE, "r") as f:
            return json.load(f)
    return {"commands": [], "aliases": {}}

def save_commands(data):
    COMMANDS_FILE.parent.mkdir(exist_ok=True)
    with open(COMMANDS_FILE, "w") as f:
        json.dump(data, f, indent=4)

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

# ─── Main App ─────────────────────────────────────────────────────────────────
class JarvisCommandManager(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("JARVIS Command Manager")
        self.geometry("900x700")
        self.configure(bg=C["bg"])
        self.resizable(True, True)
        self.minsize(750, 550)

        # Data
        self.data = load_commands()
        self.selected_index = None

        # Style
        self._setup_styles()
        self._build_ui()
        self._refresh_list()

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Dark.TFrame", background=C["bg"])
        style.configure("Card.TFrame", background=C["bg2"])
        style.configure("Dark.TLabel", background=C["bg"], foreground=C["text"],
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=C["bg"], foreground=C["accent"],
                        font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=C["bg"], foreground=C["dim"],
                        font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background=C["bg2"], foreground=C["text"],
                        font=("Segoe UI", 11, "bold"))
        style.configure("CardDim.TLabel", background=C["bg2"], foreground=C["dim"],
                        font=("Segoe UI", 9))
        style.configure("Accent.TButton", background=C["accent"], foreground="#000",
                        font=("Segoe UI", 10, "bold"), padding=(16, 8))
        style.map("Accent.TButton",
                  background=[("active", C["accent2"]), ("pressed", C["accent2"])])
        style.configure("Danger.TButton", background=C["red"], foreground="#fff",
                        font=("Segoe UI", 10), padding=(12, 6))
        style.map("Danger.TButton",
                  background=[("active", "#dc2626")])
        style.configure("Ghost.TButton", background=C["bg3"], foreground=C["text"],
                        font=("Segoe UI", 10), padding=(12, 6))
        style.map("Ghost.TButton",
                  background=[("active", C["border"])])

    def _build_ui(self):
        # ── Header ──
        header = ttk.Frame(self, style="Dark.TFrame")
        header.pack(fill="x", padx=24, pady=(20, 0))

        ttk.Label(header, text="JARVIS", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="  Command Manager", style="Subtitle.TLabel").pack(side="left", pady=(6, 0))

        btn_frame = ttk.Frame(header, style="Dark.TFrame")
        btn_frame.pack(side="right")
        ttk.Button(btn_frame, text="+ New Command", style="Accent.TButton",
                   command=self._new_command).pack(side="left", padx=(0, 8))
        ttk.Button(btn_frame, text="+ New Alias", style="Ghost.TButton",
                   command=self._new_alias).pack(side="left")

        # ── Search bar ──
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
        # Placeholder
        search_entry.bind("<FocusIn>", lambda e: None)

        # ── Command list (scrollable) ──
        list_frame = ttk.Frame(self, style="Dark.TFrame")
        list_frame.pack(fill="both", expand=True, padx=24, pady=(16, 0))

        canvas = tk.Canvas(list_frame, bg=C["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        self.scroll_inner = ttk.Frame(canvas, style="Dark.TFrame")

        self.scroll_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.scroll_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas = canvas

        # ── Aliases section ──
        alias_header = ttk.Frame(self, style="Dark.TFrame")
        alias_header.pack(fill="x", padx=24, pady=(12, 0))
        ttk.Label(alias_header, text="Aliases", style="Dark.TLabel",
                  font=("Segoe UI", 12, "bold")).pack(side="left")

        self.alias_frame = ttk.Frame(self, style="Dark.TFrame")
        self.alias_frame.pack(fill="x", padx=24, pady=(8, 0))

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Ready")
        status = tk.Label(self, textvariable=self.status_var, bg=C["bg2"],
                         fg=C["dim"], font=("Segoe UI", 9), anchor="w", padx=12, pady=6)
        status.pack(fill="x", side="bottom")

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
            lbl = tk.Label(self.scroll_inner, text="No custom commands yet. Click '+ New Command' to create one!",
                          bg=C["bg"], fg=C["dim"], font=("Segoe UI", 11))
            lbl.pack(pady=40)
        else:
            for i, cmd in enumerate(commands):
                # Filter by search
                searchable = f"{cmd.get('name','')} {' '.join(cmd.get('triggers',[]))} {cmd.get('action','')}".lower()
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
                tk.Label(row, text=f"  ->  {target}", bg=C["bg3"], fg=C["dim"],
                        font=("Segoe UI", 10)).pack(side="left")
                del_btn = tk.Button(row, text="x", bg=C["bg3"], fg=C["red"],
                                   font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2",
                                   command=lambda a=alias: self._delete_alias(a))
                del_btn.pack(side="right")

        count = len(commands)
        self.status_var.set(f"{count} command{'s' if count != 1 else ''}  |  {len(aliases)} alias{'es' if len(aliases) != 1 else ''}")

    def _build_card(self, index, cmd):
        """Build a single command card."""
        card = tk.Frame(self.scroll_inner, bg=C["bg2"], padx=16, pady=14,
                       highlightbackground=C["border"], highlightthickness=1)
        card.pack(fill="x", pady=4)

        # Top row: name + action badge
        top = tk.Frame(card, bg=C["bg2"])
        top.pack(fill="x")

        tk.Label(top, text=cmd.get("name", "Untitled"), bg=C["bg2"], fg=C["text"],
                font=("Segoe UI", 12, "bold")).pack(side="left")

        action = cmd.get("action", "")
        badge_color = C["accent3"] if not action.startswith("chain:") else C["yellow"]
        badge_text = action if len(action) < 25 else action[:22] + "..."
        badge = tk.Label(top, text=badge_text, bg=C["bg2"], fg=badge_color,
                        font=("Consolas", 9))
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

        edit_btn = tk.Button(btn_row, text="Edit", bg=C["bg3"], fg=C["text"],
                            font=("Segoe UI", 9), bd=0, padx=12, pady=4, cursor="hand2",
                            command=lambda i=index: self._edit_command(i))
        edit_btn.pack(side="left", padx=(0, 6))

        dup_btn = tk.Button(btn_row, text="Duplicate", bg=C["bg3"], fg=C["text"],
                           font=("Segoe UI", 9), bd=0, padx=12, pady=4, cursor="hand2",
                           command=lambda i=index: self._duplicate_command(i))
        dup_btn.pack(side="left", padx=(0, 6))

        del_btn = tk.Button(btn_row, text="Delete", bg=C["bg3"], fg=C["red"],
                           font=("Segoe UI", 9), bd=0, padx=12, pady=4, cursor="hand2",
                           command=lambda i=index: self._delete_command(i))
        del_btn.pack(side="left")

    # ─── Command CRUD dialogs ────────────────────────────────────────────

    def _new_command(self):
        """Open dialog to create a new command."""
        dialog = CommandDialog(self, title="New Command")
        self.wait_window(dialog)
        if dialog.result:
            self.data.setdefault("commands", []).append(dialog.result)
            save_commands(self.data)
            self._refresh_list()
            self.status_var.set(f"Created command: {dialog.result['name']}")

    def _edit_command(self, index):
        """Open dialog to edit an existing command."""
        cmd = self.data["commands"][index]
        dialog = CommandDialog(self, title="Edit Command", existing=cmd)
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


# ─── Command Edit Dialog ─────────────────────────────────────────────────────
class CommandDialog(tk.Toplevel):
    def __init__(self, parent, title="Command", existing=None):
        super().__init__(parent)
        self.title(title)
        self.geometry("550x520")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        self.result = None
        self.transient(parent)
        self.grab_set()

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
            "description": self.desc_var.get().strip()
        }
        self.destroy()


# ─── Alias Dialog ─────────────────────────────────────────────────────────────
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


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = JarvisCommandManager()
    app.mainloop()
