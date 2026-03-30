#!/usr/bin/env python3
"""
Test Claude Code Agent SDK integration without subprocess
"""

import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def test_claude_sdk_simple():
    """Simple test of Claude Code Agent SDK"""
    print("Testing Claude Code Agent SDK (no subprocess)...")
    
    try:
        options = ClaudeAgentOptions(
            system_prompt="You are JARVIS from Iron Man. Respond in 1-2 sentences with witty British humor.",
            permission_mode="auto",
            model="sonnet",
            output_format="text",
        )
        
        print("Sending test query...")
        response_text = ""
        
        async for message in query(prompt="Hello Jarvis, what's the status?", options=options):
            # Debug: print message structure
            print(f"Message type: {type(message)}")
            if hasattr(message, 'type'):
                print(f"Message.type: {message.type}")
            if hasattr(message, 'content'):
                print(f"Message.content: {message.content}")
            
            # Extract text content
            if hasattr(message, 'content') and message.content:
                if hasattr(message.content, 'text'):
                    response_text += message.content.text
                elif isinstance(message.content, str):
                    response_text += message.content
            elif hasattr(message, 'text'):
                response_text += message.text
            
            # Stop on result message
            if hasattr(message, 'type') and message.type == 'result':
                break
        
        print(f"\n✅ Claude response: {response_text}")
        return len(response_text) > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_claude_sdk_simple())
    print(f"\n🎯 Test result: {'PASSED' if success else 'FAILED'}")
