#!/usr/bin/env python3
"""
Claude vs GPT-4o-mini Cost Comparison
=====================================
Real-time pricing analysis for Jarvis AI usage
"""

import json
from datetime import datetime

# Current pricing (as of March 2026 - verified from official sources)
PRICING = {
    "claude_3_sonnet": {
        "input_per_mtok": 3.00,  # $3 per 1M input tokens
        "output_per_mtok": 15.00, # $15 per 1M output tokens
        "description": "Claude Sonnet 4 (general-purpose)"
    },
    "claude_3_haiku": {
        "input_per_mtok": 1.00,   # $1 per 1M input tokens  
        "output_per_mtok": 5.00,  # $5 per 1M output tokens
        "description": "Claude Haiku 4.5 (fast, cost-optimized)"
    },
    "claude_3_opus": {
        "input_per_mtok": 5.00,   # $5 per 1M input tokens
        "output_per_mtok": 25.00, # $25 per 1M output tokens
        "description": "Claude Opus 4.6 (flagship, most capable)"
    },
    "gpt_4o_mini": {
        "input_per_mtok": 0.15,   # $0.15 per 1M input tokens
        "output_per_mtok": 0.60,  # $0.60 per 1M output tokens
        "description": "GPT-4o-mini (OpenAI's cheapest)"
    },
    "gpt_4o": {
        "input_per_mtok": 2.50,   # $2.50 per 1M input tokens
        "output_per_mtok": 10.00, # $10 per 1M output tokens
        "description": "GPT-4o (OpenAI's balanced model)"
    },
    "gpt_5_2": {
        "input_per_mtok": 1.75,   # $1.75 per 1M input tokens
        "output_per_mtok": 14.00, # $14 per 1M output tokens
        "description": "GPT-5.2 (OpenAI's current flagship)"
    }
}

# Jarvis typical usage per request
JARVIS_USAGE = {
    "system_prompt_tokens": 2000,  # JARVIS system prompt
    "user_input_tokens": 50,       # Typical user speech
    "ai_response_tokens": 100,     # Typical AI response
    "total_input_tokens": 2050,    # System + user
    "total_output_tokens": 100,     # AI response only
}

def calculate_cost(model_name, input_tokens, output_tokens):
    """Calculate cost for a specific model."""
    pricing = PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_mtok"]
    total_cost = input_cost + output_cost
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def compare_costs():
    """Compare costs across all models for Jarvis usage."""
    print("=" * 80)
    print("💰 CLAUDE vs CHATGPT COST COMPARISON")
    print("=" * 80)
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    print("📊 JARVIS TYPICAL USAGE PER REQUEST:")
    print(f"   • System prompt: {JARVIS_USAGE['system_prompt_tokens']:,} tokens")
    print(f"   • User input: {JARVIS_USAGE['user_input_tokens']:,} tokens") 
    print(f"   • AI response: {JARVIS_USAGE['ai_response_tokens']:,} tokens")
    print(f"   • Total input: {JARVIS_USAGE['total_input_tokens']:,} tokens")
    print(f"   • Total output: {JARVIS_USAGE['total_output_tokens']:,} tokens")
    print()
    
    print("💲 COST PER REQUEST:")
    print("-" * 80)
    
    results = {}
    for model_name, pricing in PRICING.items():
        costs = calculate_cost(model_name, JARVIS_USAGE["total_input_tokens"], JARVIS_USAGE["total_output_tokens"])
        results[model_name] = costs
        
        # Format model name
        display_name = model_name.replace("_", " ").title()
        
        print(f"{display_name:<20} ${costs['total_cost']:.6f}")
        print(f"{'':20}  Input: ${costs['input_cost']:.6f} | Output: ${costs['output_cost']:.6f}")
        print(f"{'':20}  {pricing['description']}")
        print()
    
    # Sort by cost
    sorted_results = sorted(results.items(), key=lambda x: x[1]["total_cost"])
    
    print("🏆 RANKING (Cheapest to Most Expensive):")
    print("-" * 80)
    for i, (model_name, costs) in enumerate(sorted_results, 1):
        display_name = model_name.replace("_", " ").title()
        print(f"{i}. {display_name:<20} ${costs['total_cost']:.6f} per request")
    
    print()
    print("📈 COST COMPARISON MULTIPLIERS:")
    print("-" * 80)
    cheapest = sorted_results[0][1]["total_cost"]
    for model_name, costs in sorted_results:
        display_name = model_name.replace("_", " ").title()
        multiplier = costs["total_cost"] / cheapest
        print(f"{display_name:<20} {multiplier:.1f}x more expensive than cheapest")
    
    print()
    print("🎯 MONTHLY USAGE SCENARIOS:")
    print("-" * 80)
    
    scenarios = [
        ("Light use (10 requests/day)", 10),
        ("Medium use (50 requests/day)", 50), 
        ("Heavy use (200 requests/day)", 200),
        ("Power user (500 requests/day)", 500)
    ]
    
    for scenario, daily_requests in scenarios:
        monthly_requests = daily_requests * 30
        print(f"\n{scenario} ({daily_requests:,}/day, {monthly_requests:,}/month):")
        
        for model_name, costs in sorted_results[:3]:  # Show top 3 cheapest
            display_name = model_name.replace("_", " ").title()
            monthly_cost = costs["total_cost"] * monthly_requests
            print(f"   {display_name:<20} ${monthly_cost:.2f}/month")
    
    print()
    print("⚡ RECOMMENDATION FOR JARVIS:")
    print("-" * 80)
    
    cheapest_model = sorted_results[0][0]
    cheapest_cost = sorted_results[0][1]["total_cost"]
    
    if cheapest_model == "gpt_4o_mini":
        print("🥇 GPT-4o-mini is CHEAPEST for Jarvis!")
        print(f"   • Only ${cheapest_cost:.6f} per request")
        print("   • Fast response times")
        print("   • Good enough for command interpretation")
        print()
        print("🤔 But consider Claude 3.5 Haiku:")
        haiku_costs = results["claude_3_haiku"]
        print(f"   • ${haiku_costs['total_cost']:.6f} per request ({haiku_costs['total_cost']/cheapest_cost:.1f}x more)")
        print("   • Better reasoning and understanding")
        print("   • More natural JARVIS-like responses")
        print("   • Still very affordable")
    
    elif cheapest_model == "claude_3_haiku":
        print("🥇 Claude 3.5 Haiku is CHEAPEST for Jarvis!")
        print(f"   • Only ${cheapest_cost:.6f} per request")
        print("   • Better quality than GPT-4o-mini")
        print("   • More natural responses")
        print()
        print("💡 Consider Claude 3.5 Sonnet:")
        sonnet_costs = results["claude_3_sonnet"]
        print(f"   • ${sonnet_costs['total_cost']:.6f} per request ({sonnet_costs['total_cost']/cheapest_cost:.1f}x more)")
        print("   • Best reasoning and understanding")
        print("   • Most JARVIS-like personality")
        print("   • Use for complex commands only")
    
    print()
    print("🎯 CURRENT JARVIS SETUP:")
    print("-" * 80)
    print("• Primary: Claude 3.5 Sonnet (higher quality, more expensive)")
    print("• Fallback: GPT-4o-mini (cheaper, reliable)")
    print("• This gives you the best of both worlds!")
    print()
    print("💡 MONEY-SAVING TIP:")
    print("• Use Claude for complex/important commands")
    print("• Use GPT-4o-mini fallback for simple commands")
    print("• The intelligent fallback system already optimizes this!")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    compare_costs()
