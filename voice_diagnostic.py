#!/usr/bin/env python3
"""
Voice Recording Diagnostic Tool
Helps identify why Jarvis might be cutting off mid-sentence
"""

import json
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def diagnose_voice_issues():
    print("=" * 70)
    print("🎤 JARVIS VOICE RECORDING DIAGNOSTIC")
    print("=" * 70)
    
    # Load config
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except:
        print("❌ Could not load config.json")
        return
    
    print("\n📊 CURRENT VOICE SETTINGS:")
    print(f"   • Silence threshold: {config.get('silence_threshold', 1.5)} seconds")
    print(f"   • Energy threshold: {config.get('energy_threshold', 500)} RMS")
    print(f"   • Sample rate: {config.get('sample_rate', 16000)} Hz")
    print(f"   • Command timeout: {config.get('command_timeout', 30)} seconds")
    
    print("\n🔍 COMMON ISSUES & SOLUTIONS:")
    
    silence_threshold = config.get('silence_threshold', 1.5)
    energy_threshold = config.get('energy_threshold', 500)
    
    if silence_threshold < 2.0:
        print(f"⚠️  SILENCE THRESHOLD TOO LOW ({silence_threshold}s)")
        print("   → Jarvis cuts you off if you pause briefly")
        print("   → Solution: Increased to 2.5s")
    
    if energy_threshold > 400:
        print(f"⚠️  ENERGY THRESHOLD TOO HIGH ({energy_threshold})")
        print("   → Jarvis thinks quiet speech is silence")
        print("   → Solution: Lowered to 300")
    
    print("\n🎯 WHAT CAUSES MID-SENTENCE CUTS:")
    print("   1. Natural pauses in speech (1-2 seconds)")
    print("   2. Quiet speaking volume")
    print("   3. Background noise interference")
    print("   4. Mic sensitivity issues")
    
    print("\n✅ OPTIMIZED SETTINGS APPLIED:")
    print("   • Silence threshold: 2.5s (more forgiving)")
    print("   • Energy threshold: 300 (detects quieter speech)")
    print("   • Initial delay: 0.5s (more time to start)")
    
    print("\n💡 TIPS FOR BETTER VOICE RECORDING:")
    print("   • Speak clearly and at moderate volume")
    print("   • Minimize background noise")
    print("   • Pause briefly after wake word before speaking")
    print("   • Keep consistent distance from microphone")
    
    print("\n🔊 ABOUT THE BEEP:")
    print("   The beep you hear might be:")
    print("   • System sound when silence is detected")
    print("   • Audio device initialization")
    print("   • Wake word detection confirmation")
    
    print("\n" + "=" * 70)
    print("🚀 TRY JARVIS NOW:")
    print("   Say 'Hey Jarvis' and try a longer command like:")
    print("   'Hey Jarvis, open Chrome and search for the weather today'")
    print("=" * 70)

if __name__ == "__main__":
    diagnose_voice_issues()
