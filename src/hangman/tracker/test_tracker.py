"""
Test script for LLM tracker functionality.

Run this to verify the tracker works before integrating into main experiments.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Patch FIRST, before any OpenAI imports
from hangman.tracker import patch_all, LLMUsageTracker, set_current_tracker, clear_current_tracker

print("=" * 60)
print("Testing LLM Tracker")
print("=" * 60)

# Apply patches
patch_all()

# Now import OpenAI (should be patched)
try:
    from openai import OpenAI
    print("✅ OpenAI SDK imported successfully")
except ImportError:
    print("❌ OpenAI SDK not installed, cannot test")
    sys.exit(1)

# Check if API key is available
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("⚠️  OPENROUTER_API_KEY not set, using dummy key for demonstration")
    api_key = "dummy-key"

# Create a tracker
tracker = LLMUsageTracker()
set_current_tracker(tracker)

print("\n" + "=" * 60)
print("Testing OpenRouter call tracking with gpt-oss-20b")
print("=" * 60)

try:
    # Create OpenAI client pointed at OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    print("\nMaking a test API call with reasoning-capable model...")
    print("Model: openai/gpt-oss-20b")
    
    # Make a call that should trigger reasoning tokens
    # GPT-OSS models support <think> tags for reasoning
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": "What is 2+2? Think step by step."}
        ],
        max_tokens=100,
        temperature=0.3
    )
    
    print(f"\nResponse: {response.choices[0].message.content[:200]}...")
    
    # Check tracker
    usage = tracker.to_dict()
    print("\n" + "=" * 60)
    print("Tracker Results:")
    print("=" * 60)
    print(f"Total calls: {usage['total_calls']}")
    print(f"Prompt tokens: {usage['prompt_tokens']}")
    print(f"Completion tokens: {usage['completion_tokens']}")
    print(f"Reasoning tokens: {usage['reasoning_tokens']}")
    print(f"Total tokens: {usage['total_tokens']}")
    print(f"Models used: {usage['calls_by_model']}")
    
    if usage['total_calls'] > 0:
        print("\n✅ SUCCESS: Tracker captured the API call!")
        if usage['reasoning_tokens'] > 0:
            print("✅ BONUS: Reasoning tokens detected!")
        else:
            print("ℹ️  Note: No reasoning tokens (may be model-specific)")
    else:
        print("\n❌ FAILURE: Tracker did not capture the call")
        
except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    print("\nThis is expected if you don't have a valid API key.")
    print("To run a full test, set OPENROUTER_API_KEY environment variable.")
    
finally:
    clear_current_tracker()

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)

