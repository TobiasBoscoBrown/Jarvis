"""
Jarvis Command Manager
=======================
CLI + TUI for adding, editing, deleting, and testing custom voice commands.
Run: python jarvis_manager.py
"""

import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
COMMANDS_FILE = BASE_DIR / "commands" / "custom_commands.json"


def load_commands():
    if COMMANDS_FILE.exists():
        with open(COMMANDS_FILE, "r") as f:
            return json.load(f)
    return {"commands": [], "aliases": {}}


def save_commands(data):
    COMMANDS_FILE.parent.mkdir(exist_ok=True)
    with open(COMMANDS_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Saved to {COMMANDS_FILE}")


def list_commands(data):
    print("\n" + "=" * 60)
    print("📋  JARVIS CUSTOM COMMANDS")
    print("=" * 60)

    if not data["commands"]:
        print("  (no custom commands yet)")
    else:
        for i, cmd in enumerate(data["commands"], 1):
            print(f"\n  {i}. {cmd['name']}")
            print(f"     Triggers: {', '.join(cmd.get('triggers', []))}")
            print(f"     Action:   {cmd.get('action', 'N/A')}")
            print(f"     Desc:     {cmd.get('description', 'N/A')}")

    print("\n" + "-" * 60)
    print("📝  ALIASES")
    print("-" * 60)
    for alias, target in data.get("aliases", {}).items():
        print(f"  '{alias}' → '{target}'")
    print()


def add_command(data):
    print("\n➕  ADD NEW COMMAND")
    print("-" * 40)

    name = input("  Command name: ").strip()
    if not name:
        print("  ❌ Name required")
        return

    triggers_raw = input("  Trigger phrases (comma-separated): ").strip()
    triggers = [t.strip() for t in triggers_raw.split(",") if t.strip()]
    if not triggers:
        print("  ❌ At least one trigger required")
        return

    print("\n  Available action types:")
    print("    1. open_cowork          — Open Claude Cowork")
    print("    2. open_terminal        — Open Windows Terminal")
    print("    3. screenshot           — Take a screenshot")
    print("    4. open_cowork_convo    — Open a specific Cowork conversation")
    print("    5. prompt_cowork        — Send a prompt to Cowork")
    print("    6. prompt_claude_code   — Send a command to Claude Code")
    print("    7. type_text            — Type text at cursor")
    print("    8. search_web           — Search Google")
    print("    9. focus_window         — Focus a window")
    print("   10. chain:...            — Custom cc.py chain (advanced)")

    action = input("\n  Action (name or chain:...): ").strip()
    description = input("  Description: ").strip()

    cmd = {
        "name": name,
        "triggers": triggers,
        "action": action,
        "description": description or f"Custom command: {name}"
    }

    data["commands"].append(cmd)
    save_commands(data)
    print(f"\n  ✅ Command '{name}' added!")


def edit_command(data):
    if not data["commands"]:
        print("  No commands to edit.")
        return

    list_commands(data)
    try:
        idx = int(input("  Edit command # : ")) - 1
        cmd = data["commands"][idx]
    except (ValueError, IndexError):
        print("  ❌ Invalid selection")
        return

    print(f"\n  Editing: {cmd['name']}")
    print("  (press Enter to keep current value)\n")

    name = input(f"  Name [{cmd['name']}]: ").strip() or cmd["name"]
    triggers_raw = input(f"  Triggers [{', '.join(cmd.get('triggers', []))}]: ").strip()
    triggers = [t.strip() for t in triggers_raw.split(",") if t.strip()] if triggers_raw else cmd.get("triggers", [])
    action = input(f"  Action [{cmd.get('action', '')}]: ").strip() or cmd.get("action", "")
    description = input(f"  Description [{cmd.get('description', '')}]: ").strip() or cmd.get("description", "")

    data["commands"][idx] = {
        "name": name,
        "triggers": triggers,
        "action": action,
        "description": description
    }
    save_commands(data)
    print(f"\n  ✅ Command updated!")


def delete_command(data):
    if not data["commands"]:
        print("  No commands to delete.")
        return

    list_commands(data)
    try:
        idx = int(input("  Delete command # : ")) - 1
        cmd = data["commands"][idx]
    except (ValueError, IndexError):
        print("  ❌ Invalid selection")
        return

    confirm = input(f"  Delete '{cmd['name']}'? (y/n): ").strip().lower()
    if confirm == "y":
        data["commands"].pop(idx)
        save_commands(data)
        print("  ✅ Deleted!")


def add_alias(data):
    print("\n📝  ADD ALIAS")
    alias = input("  Short name (what you say): ").strip().lower()
    target = input("  Maps to (full name): ").strip()
    if alias and target:
        data.setdefault("aliases", {})[alias] = target
        save_commands(data)
        print(f"  ✅ Alias '{alias}' → '{target}' added!")


def export_commands(data):
    """Export commands for sharing or backup."""
    export_path = BASE_DIR / "commands" / f"backup_{Path(COMMANDS_FILE).stem}.json"
    with open(export_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"  ✅ Exported to {export_path}")


def main():
    print("\n" + "=" * 60)
    print("🤖  JARVIS COMMAND MANAGER")
    print("=" * 60)

    data = load_commands()

    while True:
        print("\n  Options:")
        print("    1. List all commands")
        print("    2. Add a command")
        print("    3. Edit a command")
        print("    4. Delete a command")
        print("    5. Add an alias")
        print("    6. Export/backup commands")
        print("    7. Quit")

        choice = input("\n  Choice: ").strip()

        if choice == "1":
            list_commands(data)
        elif choice == "2":
            add_command(data)
            data = load_commands()  # Reload
        elif choice == "3":
            edit_command(data)
            data = load_commands()
        elif choice == "4":
            delete_command(data)
            data = load_commands()
        elif choice == "5":
            add_alias(data)
            data = load_commands()
        elif choice == "6":
            export_commands(data)
        elif choice == "7":
            print("\n  👋 Bye!")
            break
        else:
            print("  ❌ Invalid choice")


if __name__ == "__main__":
    main()
