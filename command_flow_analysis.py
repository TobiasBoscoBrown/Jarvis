#!/usr/bin/env python3
"""
Jarvis Command Flow Analysis - How Commands Work
"""

def analyze_command_flow():
    print("=" * 80)
    print("🧠 JARVIS COMMAND FLOW ANALYSIS")
    print("=" * 80)
    
    print("\n📤 STEP 1: VOICE → TEXT")
    print("   User says: 'Hey Jarvis, launch Slay the Spire 2'")
    print("   ↓")
    print("   OpenWakeWord detects 'Hey Jarvis'")
    print("   ↓")
    print("   Audio recorded → Whisper API transcribes")
    print("   ↓")
    print("   Text: 'launch slay the spire 2'")
    
    print("\n🧠 STEP 2: TEXT → AI BRAIN")
    print("   Text goes to CommandRouter.route()")
    print("   ↓")
    print("   Checks custom commands first (fast path)")
    print("   ↓")
    print("   If no match → sends to AI (Claude/GPT-4o-mini)")
    
    print("\n📤 STEP 3: API CALL CONTENTS")
    print("   System Prompt (JARVIS_SYSTEM_PROMPT):")
    print("   ├── Persona: 'JARVIS from Iron Man, witty British humor'")
    print("   ├── Context: Current time, date")
    print("   ├── Instructions: 'You have FULL CONTROL via cc.py'")
    print("   ├── Available Actions: [ACTION: chain ...] syntax")
    print("   ├── Examples: 'open YouTube' → chain commands")
    print("   └── Format Rules: Must include [SPEAK] and [ACTION] tags")
    
    print("\n   User Prompt:")
    print("   ├── Original text: 'launch slay the spire 2'")
    print("   ├── Added context: 'User said: launch slay the spire 2'")
    print("   └── Time context: 'Current time: 3:08 AM on Sunday...'")
    
    print("\n📤 STEP 4: AI RESPONSE")
    print("   AI receives full context and responds:")
    print("   ↓")
    print("   [ACTION: chain \"launch C:\\Program Files (x86)\\Steam\\steam.exe; wait 3; type Slay the Spire 2; key enter\"]")
    print("   [SPEAK] Launching Steam and searching for Slay the Spire. Let's see if you can conquer it this time....")
    
    print("\n⚙️ STEP 5: PARSING & EXECUTION")
    print("   Response goes to _parse_and_execute():")
    print("   ↓")
    print("   Regex extracts: r'\\[ACTION:\\s*(.+?)\\]'")
    print("   Regex extracts: r'\\[SPEAK\\]\\s*(.+?)'")
    print("   ↓")
    print("   Actions found: ['chain \"launch steam.exe...\"']")
    print("   Speech found: 'Launching Steam and searching...'")
    
    print("\n🎮 STEP 6: COMMAND EXECUTION")
    print("   _execute_action() processes:")
    print("   ↓")
    print("   'chain' → _run_cc('chain', 'launch steam.exe...')")
    print("   ↓")
    print("   cc.py executes: launch → wait → type → key enter")
    print("   ↓")
    print("   Steam launches, types search, presses enter")
    
    print("\n🔊 STEP 7: VOICE RESPONSE")
    print("   speak_text sent to JarvisTTS.speak():")
    print("   ↓")
    print("   edge-tts generates audio: 'Launching Steam...'")
    print("   ↓")
    print("   Pygame plays audio instantly")
    
    print("\n" + "=" * 80)
    print("📊 WHAT WE'RE PASSING TO THE AI:")
    print("=" * 80)
    
    print("\n🧠 SYSTEM PROMPT (~2000 tokens):")
    print("   • Complete JARVIS persona and behavior")
    print("   • Full list of available actions and syntax")
    print("   • Examples of proper response format")
    print("   • Voice transcription context")
    print("   • Steam path instructions")
    
    print("\n👤 USER PROMPT (~50 tokens):")
    print("   • Time context: '3:08 AM on Sunday, March 30, 2026'")
    print("   • User speech: 'User said: launch slay the spire 2'")
    
    print("\n💰 TOTAL COST PER REQUEST:")
    print("   • Input: ~2050 tokens (system + user)")
    print("   • Output: ~100 tokens (action + speech)")
    print("   • GPT-4o-mini: ~$0.003 per request")
    print("   • Claude: ~$0.015 per request")
    
    print("\n🎯 WHY THIS WORKS:")
    print("   ✅ AI has full context of capabilities")
    print("   ✅ Structured format ensures reliable parsing")
    print("   ✅ Fallback system guarantees 100% uptime")
    print("   ✅ Voice forgiveness handles transcription errors")
    print("   ✅ All PC control through cc.py integration")

if __name__ == "__main__":
    analyze_command_flow()
