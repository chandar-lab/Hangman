## Two‑Pass Thinking → Answer: Problem and Proposed Solution

### Background
Some reasoning models emit a private trace inside <think>…</think> (or <thinking>…</thinking>) before the public answer. In practice, the model sometimes never emits the closing tag before `max_tokens`, so the unfinished thinking spills into the public answer. Today, parsing in `src/hangman/providers/llmprovider.py` treats a response without a closing tag as a direct answer, which leaks incomplete thoughts.

### Goals (desiderata)
- **No‑think fallback**: When `thinking=True` but the model does not output any think tag, treat all tokens as the public `response` and keep `thinking=""`.
- **Stable API**: Do not change the output shape of `LLMProvider.invoke`. It must still return `{ "response": str, "thinking": str }`.
- **Separate token budgets**: `LLMProvider` accepts two limits: `max_thinking_tokens` and `max_response_tokens`.
- **Conditioned final answer**: The final answer must be generated conditioned on the thinking trace, resuming the generation immediately after `</think>` with no extra user/system turns. The continuation should be a strict suffix continuation of the same assistant turn.
- **Force thought envelope**: The first generation should start with `<think>`/`<thinking>` and hard‑stop after `max_thinking_tokens` tokens at `</think>`/`</thinking>` (tag selected via `think_tag`). Then resume generation right after the closing tag for the public answer.

### Options considered
- **Option 1: Keep LangChain `ChatOpenAI` (OpenAI‑compatible HTTP)**
  - Pros: Portable, integrated with LangChain ecosystem.
  - Cons: We cannot attach token‑level `logits_processors`. The two‑pass requires adding an internal assistant message (not persisted) to represent the private trace before continuation.

- **Option 2: Switch to native vLLM Python API**
  - Pros: Exact token‑level control using a `logits_processor` to cap thinking length and force the end tag; clean continuation exactly after `</think>`.
  - Cons: Not using the LangChain `ChatOpenAI` client; requires prompt templating and direct vLLM integration.

### Decision
Implement both backends behind the existing `LLMProvider` facade:
- **`openai` backend** (default): two‑pass via `ChatOpenAI`, using stop sequences and an internal‑only assistant message for continuation. Engine/agents see the same return shape; the internal message is not logged.
- **`vllm_native` backend**: two‑pass via `vllm.LLM` + `SamplingParams`, with a custom `ThinkLogitsProcessor` to force `</think>` after `max_thinking_tokens`, and exact continuation after the tag.

### Configuration additions
Per provider in `config.yaml` under each `providers: [...]` entry:

```yaml
providers:
  - name: "qwen3_14b_local_vllm_native"
    provider_backend: openai            # one of: openai | vllm_native (default: openai)
    model_name: "Qwen/Qwen3-14B"
    parsing_format: "think_tags"
    api_config:
      base_url: "http://localhost:8000/v1"
      api_key_env: "VLLM_API_KEY"
    generation_config:
      temperature: 0.3
      max_tokens: 16384                 # legacy cap (still honored when specific caps absent)
      two_pass: true                    # when thinking=True, enable two‑pass flow
      think_tag: think                  # or: thinking
      max_thinking_tokens: 256          # cap for private trace
      max_response_tokens: 1024         # cap for public answer
```

### Provider interface (non‑breaking)
- Keep: `invoke(messages: List[BaseMessage], thinking: bool = False) -> ModelOutput`.
- Add optional kwargs (with sensible defaults from `generation_config`):
  - `max_thinking_tokens: Optional[int] = None`
  - `max_response_tokens: Optional[int] = None`
  - `think_tag: Optional[Literal["think", "thinking"]] = None`
  - `two_pass: Optional[bool] = None`  (defaults to `True` when `thinking=True`)

Return shape stays `{ "response": str, "thinking": str }`.

### Algorithms
#### Common parsing rules
- If there is an opening tag and a closing tag: extract `thinking` between them; the remainder (if any) is a `response_prefix`.
- If there is an opening tag but no closing tag: treat everything after the opening tag as `thinking`; do not leak into `response`; synthesize the closing tag for the continuation step.
- If there is no opening tag at all: treat the entire text as `response`, `thinking=""` (no‑think fallback).

#### Two‑pass for `openai` backend (ChatOpenAI)
1) **Pass 1 (thinking)**
   - Call with the normal `messages`.
   - Prompting: ensure the system prompt instructs the assistant to start the turn with `<think>` and close with `</think>`. We cannot hard‑force the first token, but we set:
     - `stop=[f"</{tag}>"]`
     - `max_tokens=max_thinking_tokens`
   - Parse per rules above. If no opening tag is present, return early with `response=first_text, thinking=""`.

2) **Pass 2 (answer continuation)**
   - Build an internal‑only assistant message: the closed thinking block `f"<{tag}>" + thinking_core + f"</{tag}>"`.
   - Call the model again with `messages + [AIMessage(content=closed_think_block)]` and `max_tokens=max_response_tokens`.
   - Do not add any system/user messages; do not log this internal assistant message to the engine.
   - Final `response = response_prefix_from_pass1 + pass2_text`.

Notes:
- This is the closest feasible approximation to “continue immediately after `</think>`” in chat APIs. True token‑level continuation is not supported by OpenAI‑compatible HTTP today.

#### Two‑pass for `vllm_native` backend
1) **Pass 1 (thinking)**
   - Build a chat‑templated prompt and append the opening tag as a prefix to the assistant completion: e.g., include `"<think>"` right at the generation boundary.
   - Use `SamplingParams` with a custom `ThinkLogitsProcessor(start_id, end_id, max_thinking_tokens)` to enforce the closing tag after the cap (or use `stop=["</think>"]`).
   - Parse per rules above.

2) **Pass 2 (answer continuation)**
   - Rebuild the prompt to end with the closed thinking block `...<think>…</think>` and continue generation normally with `max_response_tokens`.
   - Final `response = response_prefix_from_pass1 + pass2_text`.

### Implementation plan
1) **Config + types** (`src/hangman/providers/llmprovider.py`)
   - Extend `ProviderConfig` with optional `provider_backend: Literal["openai", "vllm_native"]`.
   - Extend `GenerationConfig` with `two_pass`, `think_tag`, `max_thinking_tokens`, `max_response_tokens`.

2) **Backend selection**
   - Inside `LLMProvider.__init__`, store `self.backend = config.get("provider_backend", "openai")`.
   - In `_create_client`:
     - If `backend == "openai"`: create a `ChatOpenAI` client (current behavior).
     - If `backend == "vllm_native"`: initialize `vllm.LLM` and tokenizer; keep them on `self` (return `None` for `client`).

3) **Invocation orchestration**
   - Extend `invoke(...)` signature to accept optional caps and flags. Resolve defaults from `generation_config`.
   - Branch on `self.backend`:
     - `openai` → `_invoke_openai_two_pass(...)`
     - `vllm_native` → `_invoke_vllm_two_pass(...)`
   - Preserve return shape.

4) **OpenAI path helpers**
   - `_first_pass_openai(messages, tag, max_thinking_tokens)` → `(thinking_core, response_prefix, had_tag)`
     - Call `self.client.invoke(...)` with `stop=[f"</{tag}>"]`, cap at `max_thinking_tokens`.
     - Parse per rules; synthesize closing tag if needed.
   - `_second_pass_openai(messages, closed_think_block, max_response_tokens)` → `response_text`
     - Call with `messages + [AIMessage(content=closed_think_block)]` and `max_response_tokens`.

5) **vLLM path helpers**
   - Maintain `self.vllm` and `self.tokenizer`.
   - Build prompts via chat template; for pass 1 append `"<think>"` at the assistant boundary.
   - Implement `ThinkLogitsProcessor(start_id, end_id, num_think_tokens)` and pass via `SamplingParams.logits_processors`.
   - For pass 2, append the closed think block to the assistant boundary and generate the continuation.

6) **Parsing guardrails**
   - Adjust `_parse_with_think_tags` so that an opening tag without a closing tag yields `{"response": "", "thinking": partial_think}` (prevents leakage).
   - Keep no‑think fallback as direct response (`thinking=""`).

7) **Backward compatibility**
   - If `two_pass` or specific caps are absent in config, fall back to legacy `generation_config.max_tokens` and one‑pass behavior.
   - Keep `parsing_format` behavior unchanged.

### Test plan (high‑level)
- **Unit tests** for parsing cases: complete tags, missing closing tag, no tag.
- **OpenAI backend** mock tests: verify two calls, internal AIMessage is not persisted, response concatenation works.
- **vLLM backend** integration test (optional on CI): verify `ThinkLogitsProcessor` caps thinking and continuation resumes after `</think>`.

### Known limitations and mitigations
- **OpenAI backend cannot truly force the opening tag**: mitigated by prompting and stop sequences; hard enforcement requires native vLLM.
- **Latency**: two passes cost an extra round‑trip. Use small `max_thinking_tokens` and early stops.
- **Template coupling** (vLLM): ensure chat template places the assistant boundary correctly so that adding `"<think>"` seeds the trace as the first tokens.

### Migration
- Add optional keys to `config.yaml` (defaults preserve current behavior).
- No changes required to agents, engine, or evaluation; they continue to call `LLMProvider.invoke` and receive the same shape.


