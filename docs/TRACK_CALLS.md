# LLM Call & Token Usage Tracking

## Objective

Add comprehensive metrics to trial logs for:
1. **Number of LLM calls** made during each trial
2. **Token usage breakdown**: prompt tokens, completion tokens, reasoning tokens (if available)

These metrics enable:
- Cost estimation and budgeting
- Performance analysis (calls vs. quality)
- Fair comparison between memory architectures
- Identification of inefficient agents

## Agent Architecture Analysis

### Call Transparency Spectrum

Agents vary significantly in LLM call observability:

| Agent | Complexity | Call Types | Uses LLMProvider? | Key Challenge |
|-------|-----------|------------|-------------------|---------------|
| **PrivateCoTAgent** | ⭐ Simple | 1 explicit call/turn | ✅ Fully | None - fully transparent |
| **WorkflowAgent** | ⭐⭐ Moderate | 2 explicit (responder + updater with retries) | ✅ Fully | Track both LLM providers separately |
| **ReactMemAgent** | ⭐⭐⭐ Complex | Variable (ReAct loop iterations) | ⚠️ Bypasses (`.client`) | Multi-step agent loop with tool calls |
| **Mem0Agent** | ⭐⭐⭐⭐ Opaque | 1 explicit + hidden Mem0 ops | ⚠️ Bypasses (`.client`) | Library makes hidden inference calls |
| **AMemAgent** | ⭐⭐⭐⭐ Opaque | 1 explicit + hidden A-mem ops | ❌ Direct OpenRouter | K/G/X extraction, evolution, linking |
| **LightMemAgent** | ⭐⭐⭐⭐⭐ Very Opaque | 1 explicit + 3-4 hidden pipeline ops | ❌ Direct OpenRouter | Compression, segmentation, metadata extraction |

**Critical Finding**: Only PrivateCoT and Workflow use `LLMProvider.invoke()`. All others bypass via `.client.invoke()` or direct OpenRouter SDK calls.

---

## Agent-Specific Details

### 1. PrivateCoTAgent
**Call Pattern:**
```python
# Line 94: Single call per turn
result = self.llm_provider.invoke(messages, thinking=True)
```
- **Calls/turn**: 1
- **Uses LLMProvider**: ✅ Yes - `llm_provider.invoke()`
- **Transparency**: Complete
- **Tracking**: Direct instrumentation of `LLMProvider.invoke()`

---

### 2. WorkflowAgent
**Call Pattern:**
```python
# Line 182: Response generation
result = self.responder_llm.invoke(messages, thinking=True)

# Lines 196-242: Memory update (retry loop, up to 5 attempts)
for attempt in range(1, max_attempts + 1):
    result = self.updater_llm.invoke(updater_messages)
```
- **Calls/turn**: 2-6 (1 responder + 1-5 updater with retries)
- **Uses LLMProvider**: ✅ Yes - both `responder_llm` and `updater_llm` are LLMProvider instances
- **Transparency**: Complete (both explicit)
- **Tracking**: Instrument both providers separately

---

### 3. ReactMemAgent
**Call Pattern:**
```python
# Line 68: Bind tools to underlying client
self.model = llm_provider.client.bind_tools(self.tools)

# Line 142: Direct client invocation
response_obj = self.model.invoke(prompt_messages)
```
- **Calls/turn**: Variable (typically 1-3, depends on tool usage)
- **Uses LLMProvider**: ⚠️ No - accesses `llm_provider.client` directly, bypasses `invoke()` wrapper
- **Transparency**: High (via LangChain)
- **Tracking**: Must instrument at `.client` level, not `LLMProvider.invoke()`

---

### 4. Mem0Agent
**Call Pattern:**
```python
# Line 197: HIDDEN LLM calls in Mem0 library
self.mem0.add(mem0_msgs, user_id=self.user_bucket, infer=True)
# ↑ Internally: fact extraction, deduplication, memory updates

# Line 221: EXPLICIT call for response (bypasses wrapper)
response_obj = self.llm_provider.client.invoke(messages_for_model)
```
- **Calls/turn**: 1 explicit + N hidden (Mem0 `infer=True`)
- **Uses LLMProvider**: ⚠️ No - uses `llm_provider.client` directly
- **Hidden calls**: Mem0 has separate LLM config (`mem0_config.yaml`)
- **Transparency**: Low (Mem0 operations opaque)
- **Tracking**: Must intercept both `.client` calls AND Mem0's internal LLM client

---

### 5. AMemAgent
**Call Pattern:**
```python
# Lines 60-64: A-mem uses OpenRouter DIRECTLY via OpenAI SDK
self.amem = AgenticMemorySystem(
    llm_backend="openrouter",
    llm_model="openai/gpt-oss-20b"
)
# Requires: OPENAI_BASE_URL=https://openrouter.ai/api/v1

# Line 98: HIDDEN LLM calls in A-mem library
self.amem.add_note(content=latest_user)

# Line 140: EXPLICIT call (bypasses wrapper)
response_obj = self.llm_provider.client.invoke(messages_for_model)
```
- **Calls/turn**: 1 explicit + N hidden (per-note operations)
- **Uses LLMProvider**: ❌ No - **A-mem uses OpenRouter directly** (via OpenAI SDK), response uses `.client`
- **Transparency**: Very low (A-mem pipeline + direct OpenRouter)
- **Tracking**: Must intercept OpenAI SDK calls to OpenRouter

---

### 6. LightMemAgent
**Call Pattern:**
```python
# Lines 115-153: Environment variable swapping for OpenRouter
# LightMem expects OPENAI_* but we have OPENROUTER_*
os.environ["OPENAI_API_KEY"] = os.environ.pop("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# Multiple HIDDEN LLM operations via LightMem:
# 1. Pre-compression (LLMLingua-2)
# 2. Topic segmentation (LLM-based)
# 3. Metadata extraction (structured LLM call)

# Line 244: EXPLICIT call (bypasses wrapper)
response_obj = self.llm_provider.client.invoke(messages_for_model)
```
- **Calls/turn**: 1 explicit + 3-4 hidden (compression, segmentation, metadata)
- **Uses LLMProvider**: ❌ No - **LightMem uses OpenRouter** (via OpenAI SDK + env vars), response uses `.client`
- **Transparency**: Very low (multi-stage pipeline + env var config)
- **Tracking**: Must intercept OpenAI SDK calls; complex env var handling


## Token Breakdown Taxonomy

Different backends provide different token categories:

| Provider | Available Metrics |
|----------|------------------|
| **OpenAI** | `prompt_tokens`, `completion_tokens`, `total_tokens` |
| **OpenRouter** | Same as OpenAI + `reasoning_tokens` (for o1-style models) |
| **vLLM Native** | Currently: none (server doesn't return usage) |

**Recommendation**: Normalize all to common schema:
```json
{
  "prompt_tokens": int,
  "completion_tokens": int,
  "reasoning_tokens": int | null,
  "total_tokens": int
}
```

---

## Implementation

### Step 1: Install the Tracker

The tracker is implemented in `src/hangman/tracker/llm_tracker.py` using monkey-patching.

**Key principle**: Patch the OpenAI SDK at import time to intercept all calls.

### Step 2: Integrate into run_sct_hangman.py

```python
# At the VERY TOP of run_sct_hangman.py, before any other imports:
from hangman.tracker import patch_all, LLMUsageTracker, set_current_tracker, clear_current_tracker

# Patch immediately
patch_all()

# Now import everything else
from hangman.providers.llmprovider import load_llm_provider
from hangman.engine_sct_hangman import SCTController
# ... rest of imports
```

### Step 3: Per-Trial Tracking

```python
def _run_trial_job(...):
    # Create tracker for this specific trial
    tracker = LLMUsageTracker()
    set_current_tracker(tracker)
    
    try:
        # Run trial normally
        controller = SCTController(...)
        controller.run()
        
        # Get usage stats
        usage = tracker.to_dict()
        
        # Add to trial payload
        trial_payload["llm_usage"] = usage
        
        # Write to log file
        _write_log(trial_payload)
        
    finally:
        # Clean up tracker
        clear_current_tracker()
```

### Step 4: Test Before Production

```bash
cd /home/mila/b/baldelld/scratch/hangman
python -m hangman.tracker.test_tracker
```

### Output Format

The tracker adds this to each trial log:

```json
{
  "llm_usage": {
    "total_calls": 15,
    "prompt_tokens": 3456,
    "completion_tokens": 892,
    "reasoning_tokens": 245,
    "total_tokens": 4593,
    "calls_by_model": {
      "openai/gpt-oss-20b": 12,
      "openai/gpt-oss-120b": 3
    }
  }
}
```

---

## Open Questions

1. **SCT Branching**: How to attribute tokens?
   - **Option A**: Sum all branches into main trial (simplest)
   - **Option B**: Track pre-fork vs. branches separately
   - **Recommendation**: Start with A, add granularity later if needed

2. **External memory LLMs**: Track separately?
   - Some agents use different models for memory vs. response
   - **Current approach**: All tracked together (transparent)
   - Can differentiate via `calls_by_model` breakdown

3. **Retry failures**: Count failed attempts?
   - WorkflowAgent updater retries
   - **Current approach**: Yes - all attempts counted
   - Provides visibility into retry overhead

