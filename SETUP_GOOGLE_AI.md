# 🚀 Setup Free Google AI API for Jarvis

## 📋 What You Need

1. **Google Account** (any Gmail account works)
2. **5 minutes** to get the API key

## 🔧 Step-by-Step Setup

### 1. Get Your Free API Key
1. Go to [**ai.google.dev**](https://ai.google.dev)
2. Sign in with your Google account
3. Click "**Get API Key**" in the top right
4. Generate a new API key (it's instant - no credit card needed!)
5. Copy the API key

### 2. Update Jarvis Config
1. Open `config.json` in your Jarvis folder
2. Replace `YOUR_GOOGLE_AI_API_KEY_HERE` with your actual API key:
```json
{
    "google_ai_api_key": "AIzaSyC...your-actual-key-here...xyz"
}
```

### 3. Restart Jarvis
```bash
python jarvis_core.py
```

## ✅ Benefits

- **Completely FREE** - No credit card required
- **Gemini 2.5 Flash** - Fast, intelligent responses  
- **250 requests/day** - Generous free tier
- **No Claude SDK** - Direct API calls
- **OpenAI Compatible** - Easy integration

## 🎯 What You Get

- **Speed**: Gemini 2.5 Flash is very fast
- **Intelligence**: Near-GPT-4 level performance
- **Long Context**: 1M token context window
- **Multimodal**: Can handle images, audio, video (if needed)

## 📊 Free Tier Limits

- **Gemini 2.5 Flash**: 10 RPM, 250 requests/day
- **Gemini 2.5 Pro**: 5 RPM, 100 requests/day  
- **Shared**: 250K tokens/minute cap

Perfect for Jarvis voice assistant usage! 🎉

## 🔄 Alternative Options

If you want other free alternatives:

1. **Groq** (300+ tok/s speed)
2. **OpenRouter** (community-funded free tier)
3. **Mistral AI** (excellent for code)

But Google AI Studio is the best starting point!
