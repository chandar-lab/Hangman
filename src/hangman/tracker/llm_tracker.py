"""
Global LLM call and token usage tracker using monkey-patching.

Usage:
    1. Import and patch at the very start of your main script, BEFORE any agents are loaded:
    
        from hangman.tracker import patch_all, LLMUsageTracker, set_current_tracker, get_current_tracker
        patch_all()
    
    2. In your trial loop, create and set a tracker:
    
        tracker = LLMUsageTracker()
        set_current_tracker(tracker)
        
        try:
            # Run your trial
            controller.run()
            
            # Get usage stats
            usage = tracker.to_dict()
            trial_payload["llm_usage"] = usage
        finally:
            clear_current_tracker()

This approach intercepts calls from:
- LLMProvider.invoke()
- Direct .client.invoke() calls
- External libraries (Mem0, AMem, LightMem) that use OpenAI SDK
- LangChain's ChatOpenAI

All calls funnel through openai.resources.chat.Completions.create(), which we patch.
"""

import contextvars
from typing import Dict, Any, Optional
from functools import wraps


# Thread-safe per-trial context using contextvars (supports parallel trial execution)
_trial_context = contextvars.ContextVar("llm_usage", default=None)


class LLMUsageTracker:
    """
    Per-trial LLM call and token usage tracker.
    
    Thread-safe for parallel execution via contextvars.
    """
    
    def __init__(self):
        self.total_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.total_tokens = 0
        self.calls_by_model = {}  # Track per-model breakdown
        
    def add_call(self, usage_data: Dict[str, Any], model: str = "unknown"):
        """
        Record a single API call with its usage statistics.
        
        Args:
            usage_data: Dict with keys: prompt_tokens, completion_tokens, reasoning_tokens, total_tokens
            model: Model name/identifier for per-model breakdown
        """
        self.total_calls += 1
        self.prompt_tokens += usage_data.get("prompt_tokens", 0)
        self.completion_tokens += usage_data.get("completion_tokens", 0)
        self.reasoning_tokens += usage_data.get("reasoning_tokens", 0)
        self.total_tokens += usage_data.get("total_tokens", 0)
        
        # Track per-model call count
        if model not in self.calls_by_model:
            self.calls_by_model[model] = 0
        self.calls_by_model[model] += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export tracker state as JSON-serializable dict for logging.
        
        Returns:
            Dict with all usage metrics
        """
        return {
            "total_calls": self.total_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "calls_by_model": self.calls_by_model,
        }
    
    def reset(self):
        """Reset all counters to zero."""
        self.total_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.total_tokens = 0
        self.calls_by_model.clear()


# ============================================
# Context Management
# ============================================

def get_current_tracker() -> Optional[LLMUsageTracker]:
    """
    Get the LLM usage tracker for the current execution context.
    
    Returns:
        LLMUsageTracker instance or None if no tracker is set
    """
    return _trial_context.get()


def set_current_tracker(tracker: LLMUsageTracker):
    """
    Set the LLM usage tracker for the current execution context.
    
    Args:
        tracker: LLMUsageTracker instance to use for this context
    """
    _trial_context.set(tracker)


def clear_current_tracker():
    """Clear the LLM usage tracker for the current execution context."""
    _trial_context.set(None)


# ============================================
# Monkey-patching
# ============================================

_original_create = None


def patch_openai_sdk():
    """
    Monkey-patch openai.resources.chat.Completions.create() to intercept all LLM calls.
    
    This intercepts:
    - Direct OpenAI SDK calls
    - LangChain's ChatOpenAI (uses OpenAI SDK internally)
    - ChatOpenRouter (uses OpenAI SDK with custom base_url)
    - External memory libraries (Mem0, AMem, LightMem) that use OpenAI SDK
    
    CRITICAL: Call this ONCE at the very start of your script, BEFORE any agents
    or LLM providers are instantiated.
    """
    global _original_create
    
    # Guard against double-patching
    if _original_create is not None:
        print("⚠️  OpenAI SDK already patched, skipping")
        return
    
    try:
        from openai.resources.chat import Completions
        
        # Save original method
        _original_create = Completions.create
        
        @wraps(_original_create)
        def tracked_create(self, *args, **kwargs):
            """Wrapper that tracks usage before returning response."""
            # Call original create method
            response = _original_create(self, *args, **kwargs)
            
            # Extract and record usage if tracker is active
            tracker = get_current_tracker()
            if tracker and hasattr(response, 'usage') and response.usage:
                usage_dict = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "reasoning_tokens": getattr(response.usage, 'reasoning_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0),
                }
                model = kwargs.get('model', 'unknown')
                tracker.add_call(usage_dict, model=model)
            
            return response
        
        # Replace original with tracked version
        Completions.create = tracked_create
        print("✅ OpenAI SDK patched for LLM call tracking")
        
    except ImportError as e:
        print(f"⚠️  OpenAI SDK not found, skipping patch: {e}")
    except Exception as e:
        print(f"❌ Failed to patch OpenAI SDK: {e}")


def patch_all():
    """
    Convenience function to apply all available patches.
    
    Currently only patches OpenAI SDK since vLLM is not used.
    Call this once at the start of your main script.
    """
    patch_openai_sdk()


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print(__doc__)
    print("\nExample integration:")
    print("""
    # At the top of run_sct_hangman.py:
    from hangman.tracker import patch_all, LLMUsageTracker, set_current_tracker, get_current_tracker
    
    # Patch before any agent imports
    patch_all()
    
    # In your trial execution function:
    def run_trial(...):
        tracker = LLMUsageTracker()
        set_current_tracker(tracker)
        
        try:
            # Run trial
            controller = SCTController(...)
            controller.run()
            
            # Get usage stats
            usage = tracker.to_dict()
            print(f"Trial used {usage['total_calls']} calls, {usage['total_tokens']} tokens")
            
            # Add to trial log
            trial_payload["llm_usage"] = usage
            
        finally:
            clear_current_tracker()
    """)

