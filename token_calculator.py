#!/usr/bin/env python3
"""
Jarvis Token Usage Calculator
Estimates monthly API costs based on typical usage patterns
"""

import tiktoken

def estimate_tokens(text):
    """Estimate token count using GPT-4 tokenizer"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimate (1 token ≈ 4 characters for English)
        return len(text) // 4

# Jarvis system prompt (roughly 280 lines)
JARVIS_SYSTEM_PROMPT = """You are JARVIS — the AI from Iron Man. Witty, dry, refined British humor.
You take subtle jabs at the user sometimes. You're helpful but never boring.
Keep spoken responses to 1-3 sentences MAX (this gets read aloud by TTS).
NEVER use markdown, bullet points, code blocks, asterisks, or special formatting.
Plain conversational English only. Address the user as "sir" sometimes but not always.

You have FULL CONTROL of the user's Windows PC through a tool called cc.py.
When the user asks you to DO something on their computer, you MUST return action commands.

=== AVAILABLE ACTIONS ===
You can return one or more [ACTION] lines. Each will be executed in order.

[ACTION: chain "command1; command2; command3"]
  Available chain commands (semicolon-separated):
  - launch <url_or_program>     → open a URL in browser or launch an app
  - focus <window_title>        → bring a window to the foreground
  - click <x> <y>              → click at pixel coordinates
  - click_text <text>          → find text on screen and click it
  - doubleclick_text <text>    → double-click text on screen
  - rightclick_text <text>     → right-click text on screen
  - type <text>                → type text at cursor
  - key <key>                  → press a key (enter, backspace, tab, escape, f5, ctrl+c, ctrl+v, alt+f4, etc.)
  - hold <modifier> key <key>  → hold modifier and press key (e.g., hold ctrl key a)
  - scroll <x> <y> <amount>   → scroll at position (negative = down, positive = up)
  - screenshot                 → take a screenshot
  - wait <seconds>             → pause between steps
  - ocr                        → read all text on screen

[ACTION: claude_code "prompt here"]
  Send a complex coding/technical task to Claude Code CLI for execution.

[ACTION: speak_only]
  Use this when you just want to talk and don't need to do anything on the PC.

=== RESPONSE FORMAT ===
Always include EXACTLY ONE [SPEAK] line with what to say aloud.
Include [ACTION] lines ONLY if you need to do something on the PC.

Examples:

User: "open YouTube"
[ACTION: chain "launch https://www.youtube.com; wait 2"]
[SPEAK] Opening YouTube for you, sir.

User: "what time is it"
[ACTION: speak_only]
[SPEAK] It's currently {time_hint}. Though I suspect you have a clock somewhere nearby.

User: "close this window"
[ACTION: chain "key alt+f4"]
[SPEAK] Window closed. Hopefully it wasn't anything important.

=== IMPORTANT RULE ===
- For time/date questions: use the time_hint provided, don't say you can't tell time
- For opening apps/sites: use [ACTION: chain "launch ..."]
- For keyboard shortcuts: use [ACTION: chain "key ..."]
- For complex multi-step PC tasks: chain multiple commands with semicolons and waits
- For coding tasks or anything needing Claude's intelligence: use [ACTION: claude_code "..."]
- For pure conversation with no PC action needed: use [ACTION: speak_only]
- ALWAYS include [SPEAK] — you must always respond verbally
- Never explain the action format to the user — just do it and talk naturally
"""

# Pricing (2026 rates)
GPT4O_MINI_INPUT_COST = 0.15 / 1_000_000  # $0.15 per million input tokens
GPT4O_MINI_OUTPUT_COST = 0.60 / 1_000_000  # $0.60 per million output tokens

def calculate_costs():
    print("=" * 70)
    print("JARVIS TOKEN USAGE & COST CALCULATOR")
    print("=" * 70)
    
    # Calculate system prompt tokens
    system_tokens = estimate_tokens(JARVIS_SYSTEM_PROMPT)
    print(f"\n📝 System Prompt: {system_tokens:,} tokens (sent with every request)")
    
    # Typical usage scenarios
    scenarios = {
        "Light User": {
            "requests_per_day": 20,
            "avg_user_input": 15,  # words
            "avg_jarvis_response": 25,  # words  
            "days_per_month": 30
        },
        "Moderate User": {
            "requests_per_day": 50,
            "avg_user_input": 20,
            "avg_jarvis_response": 30,
            "days_per_month": 30
        },
        "Heavy User": {
            "requests_per_day": 100,
            "avg_user_input": 25,
            "avg_jarvis_response": 35,
            "days_per_month": 30
        },
        "Power User": {
            "requests_per_day": 200,
            "avg_user_input": 30,
            "avg_jarvis_response": 40,
            "days_per_month": 30
        }
    }
    
    print(f"\n💰 Pricing (GPT-4o-mini):")
    print(f"   Input: ${GPT4O_MINI_INPUT_COST:.6f} per token (${GPT4O_MINI_INPUT_COST*1_000_000:.2f}/M)")
    print(f"   Output: ${GPT4O_MINI_OUTPUT_COST:.6f} per token (${GPT4O_MINI_OUTPUT_COST*1_000_000:.2f}/M)")
    
    for scenario_name, config in scenarios.items():
        print(f"\n--- {scenario_name.upper()} ---")
        
        # Estimate tokens (rough: 1 token ≈ 0.75 words for English)
        input_tokens_per_request = system_tokens + (config["avg_user_input"] / 0.75)
        output_tokens_per_request = config["avg_jarvis_response"] / 0.75
        
        # Monthly calculations
        requests_per_month = config["requests_per_day"] * config["days_per_month"]
        monthly_input_tokens = input_tokens_per_request * requests_per_month
        monthly_output_tokens = output_tokens_per_request * requests_per_month
        
        # Costs
        monthly_input_cost = monthly_input_tokens * GPT4O_MINI_INPUT_COST
        monthly_output_cost = monthly_output_tokens * GPT4O_MINI_OUTPUT_COST
        total_monthly_cost = monthly_input_cost + monthly_output_cost
        
        print(f"   Requests: {config['requests_per_day']}/day × {config['days_per_month']} days = {requests_per_month:,}/month")
        print(f"   Tokens per request: {input_tokens_per_request:.0f} in + {output_tokens_per_request:.0f} out")
        print(f"   Monthly tokens: {monthly_input_tokens:,.0f} in + {monthly_output_tokens:,.0f} out")
        print(f"   Monthly cost: ${monthly_input_cost:.2f} + ${monthly_output_cost:.2f} = ${total_monthly_cost:.2f}")
        
        # Annual projection
        annual_cost = total_monthly_cost * 12
        print(f"   Annual cost: ${annual_cost:.2f}")
    
    print(f"\n" + "=" * 70)
    print("💡 COST OPTIMIZATION TIPS:")
    print("   • GPT-4o-mini is ~100x cheaper than GPT-4o")
    print("   • System prompt is ~{system_tokens:,} tokens - consider trimming if needed")
    print("   • Average voice interaction costs ~${(system_tokens + 40/0.75 + 50/0.75) * GPT4O_MINI_INPUT_COST + (50/0.75) * GPT4O_MINI_OUTPUT_COST:.4f} per request")
    print("   • Even heavy users pay less than $20/month")
    print("=" * 70)

if __name__ == "__main__":
    calculate_costs()
