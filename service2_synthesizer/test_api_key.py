"""
test_api_key.py
Run from service2_synthesizer/ with venv active:
    python test_api_key.py

Tests the API key and model configured in .env.
Sends a minimal single-token request — costs almost nothing.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
model    = os.getenv("LLM_MODEL", "")

print(f"Provider : {provider}")
print(f"Model    : {model}")
print(f"Testing  ...\n")

try:
    if provider == "anthropic":
        import anthropic
        key = os.getenv("ANTHROPIC_API_KEY", "")
        print(f"Key      : {key[:8]}{'*' * (len(key) - 8) if len(key) > 8 else '(too short)'}")
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
        )
        print(f"Response : {response.content[0].text.strip()}")
        print("\n✓ Anthropic API key is valid and working.")

    elif provider in ("openai", "openai-compatible"):
        from openai import OpenAI
        key      = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL") or None
        print(f"Key      : {key[:8]}{'*' * (len(key) - 8) if len(key) > 8 else '(too short)'}")
        if base_url:
            print(f"Base URL : {base_url}")
        client = OpenAI(api_key=key, **({"base_url": base_url} if base_url else {}))
        response = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            max_tokens=10,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
        )
        print(f"Response : {response.choices[0].message.content.strip()}")
        print("\n✓ OpenAI API key is valid and working.")

    elif provider == "google":
        import google.generativeai as genai
        key = os.getenv("GOOGLE_API_KEY", "")
        print(f"Key      : {key[:8]}{'*' * (len(key) - 8) if len(key) > 8 else '(too short)'}")
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model or "gemini-1.5-flash")
        response = m.generate_content("Reply with just: OK")
        print(f"Response : {response.text.strip()}")
        print("\n✓ Google API key is valid and working.")

    elif provider == "ollama":
        import ollama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"Base URL : {base_url}")
        response = ollama.chat(
            model=model or "llama3",
            messages=[{"role": "user", "content": "Reply with just: OK"}],
        )
        print(f"Response : {response['message']['content'].strip()}")
        print("\n✓ Ollama is reachable and working.")

    else:
        print(f"✗ Unknown provider: {provider}")
        print("  Set LLM_PROVIDER in .env to one of:")
        print("  anthropic | openai | openai-compatible | google | ollama")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ Failed: {e}")
    print("\nCommon causes:")
    print("  - Wrong or missing API key in .env")
    print("  - Wrong model name for this provider")
    print("  - No internet connection")
    print("  - Ollama not running (for ollama provider)")
    sys.exit(1)
