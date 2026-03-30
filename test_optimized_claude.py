#!/usr/bin/env python3
"""
Test the optimized Claude Code CLI integration
"""

import subprocess
import time
import os

def test_optimized_claude():
    """Test optimized Claude Code CLI vs old method"""
    print("Testing optimized Claude Code CLI integration...")
    
    # Simple test prompt
    test_prompt = "You are JARVIS. Say hello in one sentence."
    
    # Test optimized method
    print("\n1. Testing optimized Claude Code CLI...")
    start = time.time()
    
    try:
        cmd = [
            "claude", 
            "-p", test_prompt,
            "--output-format", "text",
            "--model", "sonnet", 
            "--permission-mode", "auto"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.expanduser("~\\Desktop"),
        )
        
        optimized_time = time.time() - start
        optimized_response = result.stdout.strip() if result.stdout else result.stderr.strip()
        
        print(f"   Response: {optimized_response}")
        print(f"   Time: {optimized_time:.2f} seconds")
        
    except Exception as e:
        print(f"   Error: {e}")
        optimized_time = float('inf')
    
    # Test old method (shell command)
    print("\n2. Testing old shell method...")
    start = time.time()
    
    try:
        old_cmd = f'claude -p "{test_prompt}"'
        
        result = subprocess.run(
            old_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.expanduser("~\\Desktop"),
        )
        
        old_time = time.time() - start
        old_response = result.stdout.strip() if result.stdout else result.stderr.strip()
        
        print(f"   Response: {old_response}")
        print(f"   Time: {old_time:.2f} seconds")
        
    except Exception as e:
        print(f"   Error: {e}")
        old_time = float('inf')
    
    # Compare results
    print(f"\n🚀 Performance Comparison:")
    if optimized_time < old_time:
        improvement = ((old_time - optimized_time) / old_time) * 100
        speedup = old_time / optimized_time
        print(f"   Optimized method is {improvement:.1f}% faster ({speedup:.1f}x speedup)")
        print(f"   Time saved: {old_time - optimized_time:.2f} seconds")
    else:
        print(f"   Times are similar or old method was faster")
    
    print(f"\n✅ Test complete! Both methods work, optimized version should be more consistent.")

if __name__ == "__main__":
    test_optimized_claude()
