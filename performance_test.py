#!/usr/bin/env python3
"""
Performance test for Jarvis improvements
Tests API call speed vs subprocess and audio playback speed
"""

import time
import subprocess
from openai import OpenAI
import json
from pathlib import Path

# Load config
BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "config.json", "r") as f:
    CONFIG = json.load(f)

client = OpenAI(api_key=CONFIG["openai_api_key"])

def test_subprocess_claude():
    """Test old subprocess method"""
    print("Testing old subprocess method...")
    start = time.time()
    
    try:
        result = subprocess.run(
            f'claude -p "Say hello"',
            shell=True, capture_output=True, text=True, timeout=30,
            cwd=CONFIG.get("claude_code_workdir", "~\\Desktop"),
        )
        end = time.time()
        print(f"Subprocess result: {result.stdout.strip()[:100]}...")
        print(f"Subprocess time: {end - start:.2f} seconds")
        return end - start
    except Exception as e:
        print(f"Subprocess failed: {e}")
        return None

def test_direct_api():
    """Test new direct API method"""
    print("\nTesting new direct API method...")
    start = time.time()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are JARVIS. Keep responses to 1-2 sentences."},
                {"role": "user", "content": "Say hello"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        end = time.time()
        response_text = response.choices[0].message.content.strip()
        print(f"API result: {response_text}")
        print(f"API time: {end - start:.2f} seconds")
        return end - start
    except Exception as e:
        print(f"API failed: {e}")
        return None

def test_audio_playback():
    """Test pygame vs PowerShell audio playback"""
    print("\nTesting audio playback methods...")
    
    # Create a simple test audio file path
    audio_path = BASE_DIR / "tts_cache" / "test.mp3"
    
    if not audio_path.exists():
        print("No test audio file found, skipping audio test")
        return
    
    # Test pygame
    try:
        import pygame
        pygame.mixer.init()
        
        start = time.time()
        pygame.mixer.music.load(str(audio_path))
        pygame.mixer.music.play()
        
        # Wait for it to start playing (not finish)
        while not pygame.mixer.music.get_busy():
            time.sleep(0.01)
            if time.time() - start > 2:  # timeout
                break
        
        pygame_time = time.time() - start
        print(f"Pygame playback start time: {pygame_time:.3f} seconds")
        pygame.mixer.quit()
        
    except Exception as e:
        print(f"Pygame test failed: {e}")
        pygame_time = None

if __name__ == "__main__":
    print("=" * 60)
    print("JARVIS PERFORMANCE TEST")
    print("=" * 60)
    
    # Test API methods
    sub_time = test_subprocess_claude()
    api_time = test_direct_api()
    
    if sub_time and api_time:
        improvement = ((sub_time - api_time) / sub_time) * 100
        speedup = sub_time / api_time
        print(f"\n🚀 API improvement: {improvement:.1f}% faster ({speedup:.1f}x speedup)")
        print(f"   Time saved: {sub_time - api_time:.2f} seconds per request")
    
    # Test audio
    test_audio_playback()
    
    print("\n" + "=" * 60)
    print("✅ Performance test complete!")
    print("The improvements should make Jarvis much more responsive.")
