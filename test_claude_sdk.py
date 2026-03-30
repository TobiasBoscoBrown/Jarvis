#!/usr/bin/env python3
"""
Test Claude Code Agent SDK integration
"""

import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def test_claude_sdk():
    """Test Claude Code Agent SDK basic functionality"""
    print("Testing Claude Code Agent SDK...")
    
    try:
        options = ClaudeAgentOptions(
            system_prompt="You are JARVIS from Iron Man. Respond in 1-2 sentences.",
            permission_mode="auto",
            model="sonnet",
            output_format="text",
        )
        
        print("Sending test query to Claude...")
        response_text = ""
        
        async for message in query(prompt="Hello Jarvis, what time is it?", options=options):
            if hasattr(message, 'content') and message.content:
                if hasattr(message.content, 'text'):
                    response_text += message.content.text
                elif isinstance(message.content, str):
                    response_text += message.content
                elif hasattr(message, 'text'):
                    response_text += message.text
            
            if hasattr(message, 'type') and message.type == 'result':
                break
        
        print(f"Claude response: {response_text}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_claude_sdk())
    if success:
        print("✅ Claude Code SDK test passed!")
    else:
        print("❌ Claude Code SDK test failed!")
