## Providers: two-pass control, classical OpenAI behavior, and tool-calling

This document captures the updated goals and a concrete plan to support:
- Classical LangChain/OpenAI behavior on the OpenAI-compatible backend (single-pass; native tool binding)
- A vLLM native backend that preserves our two-pass thinking control
- Multiple tool-parsing strategies, including Qwen3 Hermes-style tool calls


### Desiderata

- **Two-pass thinking (only on vLLM native backend)**
  - Cap the private reasoning length with a first pass that stops at </think> or </thinking> and a configurable token budget.
  - Produce a closed think block and continue the public answer in a second pass.
  - Preserve the think block for logging/trace; attach it to the returned message metadata.

- **Classical OpenAI path (no two-pass on OpenAI backend)**
  - Use the standard LangChain `ChatOpenAI` logic and its built-in `.bind_tools(...)` and `.invoke(...)` behavior.
  - Keep the current `providers/llmprovider.py` behavior: single-pass generation, optional think-tag parsing on the client side, and tool calling via native OpenAI/LangChain bindings.

- **Tool-calling across backends**
  - For OpenAI backend: continue relying on `ChatOpenAI.bind_tools(...)` and native tool calling (classical path).
  - For vLLM native backend: add tool-calling support. Client can bind tools and invoke the server with tool schema; server returns structured tool calls.

- **Multiple tool-parsing strategies**
  - Support an explicit `tool_parser` setting, at least:
    - `openai` (native OpenAI tool-calls; use `.bind_tools`)
    - `hermes` (Qwen-style parser). Hermes requires the tool list and call format to be injected into the prompt, e.g.:
      - Append to the system message a Tools section and a `<tools>...</tools>` JSON with each function schema.
      - Instruct the model to return tool calls as a `<tool_call>{...}</tool_call>` JSON object.
  - The function descriptions come from the tool docstrings and are appended by the client when `tool_parser=hermes` is active.

- **Stable message semantics for agents**
  - `reactmem_agent.py` calls `main_llm_provider.client.bind_tools(self.tools)` and expects `.invoke(...)` to return a LangChain `AIMessage` with:
    - `.content` string
    - `.tool_calls` list with items shaped like `{id, name, args}`
    - `additional_kwargs["thinking"]` containing the private trace
  - This must work identically for both backends.

- **Config-driven**
  - Keep configuration in `config.yaml` per provider:
    - `provider_backend`: `openai` | `vllm_native`
    - `parsing_format`: `think_tags` | `direct_response`
    - `tool_parser`: `openai` | `hermes`
    - `generation_config`: temperature, max_tokens, and for vLLM native also `two_pass`, `think_tag`, `max_thinking_tokens`, `max_response_tokens`

- **Robustness and safety**
  - Graceful behavior on unterminated think tags: preserve a synthesized closing and surface a safe public error when needed (vLLM native path), or treat as direct response otherwise.
  - Allow empty API key on localhost.
  - Clear errors and timeouts; never crash the agent.


## Design choices

### One provider API; OpenAI uses ChatOpenAI directly, vLLM native uses a custom client

Expose a uniform `LLMProvider.client` that is always present and LangChain-compatible, with:
- `bind_tools(self, tools, tool_choice="auto") -> client`
- `invoke(self, messages) -> AIMessage`

Backends:
- **OpenAI backend (classical path):** use `langchain_openai.ChatOpenAI` directly (no custom wrapper).
  - Single-pass only.
  - `tool_parser=openai`: use native `.bind_tools(...)` so tools are sent in OpenAI schema; `.invoke(...)` returns `AIMessage` as usual.
  - `tool_parser=hermes`: rely on the server’s Hermes parser (e.g., vLLM OpenAI endpoint started with `--tool-call-parser hermes`). Client still calls `.bind_tools(...)`; the server performs Hermes formatting. No client-side Hermes injection.

- **ChatVllm (two-pass path):** custom LangChain-compatible client that talks to our `/generate` endpoint.
  - Always two-pass when configured (`generation_config.two_pass: true`).
  - Accepts `tools` and `tool_choice`; forwards them and `tool_parser` to the server.
  - Returns an `AIMessage` with `.content`, `.tool_calls`, and `additional_kwargs["thinking"]`.
  - For `tool_parser=hermes`: the server constructs the Hermes preamble and extracts `<tool_call>` JSON.

Notes:
- LLMProvider responsibilities vs client responsibilities:
  - `LLMProvider.invoke(...)` returns a dict `{response, thinking}` and is used by simpler agents (e.g., `WorkflowAgent`).
  - `LLMProvider.client` is a LangChain-compatible chat client (either `ChatOpenAI` or `ChatVllm`) used by `ReActMemAgent`.
  - `client.invoke(...)` MUST emulate `ChatOpenAI`: return an `AIMessage` with `.content` (raw text) and `.tool_calls` (if any). The client does NOT split response/thinking.
  - The agent will call `LLMProvider.parse_response(response_obj.content)` inside `_call_model` to obtain `{response, thinking}`. For final messages without tools, the agent will overwrite `response_obj.content` with the parsed `response`.
- We keep think-tag parsing in `LLMProvider.parse_response(...)`. The client (`ChatVllm`) should return the raw combined text (closed think block + public text) in `.content`, letting the agent parse it.
- We avoid adding two-pass to the OpenAI backend to maintain classical behavior and compatibility with OpenAI/OpenRouter tool semantics and streaming.


### Hermes tool formatting and parsing

When `tool_parser=hermes`, the client (or server for vLLM native) will:
- Add to the system message after the base prompt:
  - A `# Tools` header and an explanatory preamble
  - A `<tools> ... </tools>` block containing a JSON array (or objects) with each function signature and JSONSchema
  - An instruction to emit tool calls inside a `<tool_call> ... </tool_call>` block containing a JSON object: `{"name": <function-name>, "arguments": <args-json-object>}` (or an array for multiple calls)
- At response time, extract the JSON inside `<tool_call> ... </tool_call>` and map it to `AIMessage.tool_calls`.

This matches the example Qwen3 Hermes formatting, where the function descriptions are appended to the system message by the parser/client and the model returns tool calls wrapped in `<tool_call>` tags.


### Server capability: vLLM native `/generate`

Extend the native server to accept tool-calling parameters and Hermes parsing:
- Request accepts:
  - `messages: List[{role, content}]`
  - `temperature`
  - `two_pass: bool`
  - `think_tag: "think" | "thinking"`
  - `max_thinking_tokens: int`, `max_response_tokens: int`
  - `tools: List[{name, description, parameters(jsonschema)}]` (optional)
  - `tool_choice: "auto" | "none" | {"name": str}` (optional)
  - `tool_parser: "openai" | "hermes"` (optional; default `hermes` for Qwen)
- Server behavior:
  - If `tool_parser=hermes`, prepend the Hermes tool preamble to the system message and instruct the model to emit `<tool_call>` JSON when it chooses a tool.
  - Two-pass thinking:
    - Pass 1: generate thinking up to `</think>`/`</thinking>` or `max_thinking_tokens` (stop sequences), capture `thinking_core` and any `response_prefix`.
    - Pass 2: append the closed think block and continue for up to `max_response_tokens`.
  - Extract tool calls:
    - If `<tool_call> ... </tool_call>` is present in the second-pass text, parse JSON, validate against provided schemas (basic type/required checks), and return `tool_calls`.
    - Otherwise, return `tool_calls: []` and the normal assistant `response` text.
- Response includes:
  - `thinking: str`
  - `response: str` (public answer; empty or minimal if tool-only output)
  - `tool_calls: Optional[List[{id, name, arguments}]]`
  - `pre_tool_text: Optional[str]` (any non-JSON assistant text preceding tool calls, if present)


## Implementation plan

1) **Introduce ChatVllm** (`src/hangman/providers/base_client.py` and `src/hangman/providers/vllm_client.py`)
   - `BaseClient` with `.bind_tools(...)` and `.invoke(...)` signatures.
   - `ChatVllm` only:
     - Store tools and `tool_choice` from `.bind_tools(...)`.
     - In `.invoke(...)`, POST to `/generate` with two-pass params and tool fields; parse JSON into `AIMessage` with `.tool_calls` and `additional_kwargs["thinking"]`.

2) **Wire provider** (`src/hangman/providers/llmprovider_new.py` or consolidate into `llmprovider.py`)
   - For `provider_backend: openai`, continue using `ChatOpenAI` exactly as in `llmprovider.py` (no wrapping, native `.bind_tools`).
   - For `provider_backend: vllm_native`, return the new `ChatVllm`.
   - Keep `parse_response(...)` unchanged for think-tag parsing in agents that strip content on final messages.

3) **Extend native server** (`src/hangman/providers/vllm_http_server.py`)
   - Request model additions:
     - `tools: List[{name: str, description: str, parameters: dict}]` (JSONSchema in `parameters`)
     - `tool_choice: Literal["auto","none"] | {"name": str}` (default: "auto")
     - `tool_parser: Literal["openai","hermes"]` (default: `"hermes"` for Qwen)
   - Prompt assembly (Hermes):
     - If `tool_parser == "hermes"`, append to the system message:
       - A Tools preamble explaining when/how to call tools
       - A `<tools> ... </tools>` block listing each function with `name`, `description`, and `parameters` JSONSchema
       - An instruction to emit tool calls inside `<tool_call> ... </tool_call>` as a JSON object:
         `{ "name": <function-name>, "arguments": <args-json-object> }`
         and to support multiple calls by returning either multiple `<tool_call>` blocks or a single JSON array inside one block
     - If `tool_choice == "none"`, add: "Do not call any tools; reply normally."
     - If `tool_choice == {"name": X}` add: "If you call a tool, it MUST be `<X>`; otherwise reply normally."
   - Two-pass generation with tool-aware stopping (Hermes):
     - Pass 1 (thinking):
       - Use the chat template with system+messages; append the opening think tag (`<think>` or `<thinking>`)
       - Sampling params: `stop=["</think>","</thinking>"]`, `max_tokens=max_thinking_tokens`
       - Extract `thinking_core` and any `response_prefix` that follows a closed tag
     - Pass 2 (answer/tool selection):
       - Build `closed_think_block = f"<{tag}>{thinking_core}</{tag}>"`
       - Append `closed_think_block` to the base chat prompt
       - In the Hermes preamble, instruct the model to place all tool calls (one or multiple) inside a SINGLE `<tool_call> ... </tool_call>` block. If multiple calls are needed, return a JSON array inside that single block.
       - If `tool_parser == "hermes"`, set stop sequences for Pass 2 to include `"</tool_call>"` so the generation halts immediately after the first (and only) closing tag of the single block. Also include a `max_response_tokens` cap.
       - Generate second-pass text `text2`
   - Post-processing and extraction:
     - Let `combined = (response_prefix + (" " if response_prefix and text2 else "") + text2).strip()`
     - Extract the `<tool_call> ... </tool_call>` block (at most one by construction) from `combined`:
       - Inside the block, accept either a single JSON object or a JSON array of calls
       - Tolerate whitespace/newlines and optional code-fence wrappers
       - Parse with `json.loads`; if parsing fails, attempt to find the first `{...}` or `[...]` inside the block and retry
     - Build `tool_calls: List[Dict]`:
       - For each parsed call, produce `{ "id": f"call_{i+1}", "name": str, "arguments": dict }`
       - If the block is an array, expand into multiple calls
     - Validate `arguments` against provided JSONSchemas (basic checks):
       - Ensure required properties exist and types are compatible (str/int/number/bool/object/array)
       - On validation failure, drop the offending call and record a warning internally (optional)
     - Determine `pre_tool_text`:
       - Remove the `<tool_call> ... </tool_call>` span from `combined`; the remaining text becomes `pre_tool_text`
       - This will be returned as `response` and also used to construct the `AIMessage.content` downstream
   - Response construction:
     - Always include:
       - `thinking: str` (the `thinking_core` from Pass 1)
       - `tool_calls: List` (possibly empty)
       - `response: str`
         - If `tool_calls` non-empty: set to `pre_tool_text` (may be empty)
         - If no tool calls: set to the full assistant text (`combined`)
     - The client (`ChatVllm`) will set the final `AIMessage.content` to the raw combined text (closed think block + `pre_tool_text` for tool-calls, or full `combined` when no tools). It will not split `{response, thinking}`.
   - Edge cases and safeguards:
     - Unterminated `<tool_call>`: if an opening tag is found without a closing tag and stop wasn’t hit, treat as incomplete; return `tool_calls: []` and a safe error message in `response`
     - Multiple tool-call blocks: parse sequential, non-overlapping blocks in order of appearance
     - Token cap: if Pass 2 hits `max_response_tokens` without `</tool_call>`, attempt JSON extraction from the partial text; if valid JSON found, accept; else treat as no-call
     - Size limits: cap the size of parsed JSON to prevent abuse; reject overly large arguments objects
   - API schemas:
     - Update `GenerateRequest` with the fields above
     - Update `GenerateResponse`:
       - `thinking: str`
       - `response: str`
       - `tool_calls: Optional[List[Dict[str, Any]]] = None`
       - `pre_tool_text: Optional[str] = None`
   - Example response (tool call made):
     - Server JSON (GenerateResponse):
       - `{"thinking": "...", "response": "I'll update memory.", "tool_calls": [{"id":"call_1","name":"overwrite_memory","arguments":{"new_memory":"..."}}], "pre_tool_text": "I'll update memory."}`
     - ChatVllm client return (LangChain AIMessage):
       - `.content`: `"<think>…</think>\nI'll update memory."` (closed think block + pre_tool_text)
       - `.tool_calls`: `[{"id":"call_1","name":"overwrite_memory","args":{"new_memory":"..."}}]`

4) **Configuration** (`config.yaml`)
   - Ensure provider names are unique (remove duplicates of `qwen3_14b_local_vllm_native`).
   - For OpenAI backend entries:
     - `provider_backend: openai`
     - `tool_parser: openai` (or `hermes` if your server is configured with the Hermes parser)
     - No `two_pass` entries (single-pass behavior).
   - For vLLM native entries:
     - `provider_backend: vllm_native`
     - `tool_parser: hermes` (default for Qwen3)
     - `generation_config`: `two_pass: true`, `think_tag`, `max_thinking_tokens`, `max_response_tokens`.

5) **Agent compatibility**
   - No edits to `src/hangman/agents/reactmem_agent.py` or `workflow_agent.py`.
   - `reactmem_agent` will keep using `provider.client.bind_tools(...)` and expect `.invoke(...)` to yield a LangChain `AIMessage` with valid `.tool_calls`.
   - The private thinking trace is available via `additional_kwargs["thinking"]` and/or through existing `parse_response(...)` when relevant.

6) **Testing**
   - Unit tests:
     - Think-tag parsing: normal and unterminated tags
     - Hermes prompt injection (client fallback) and parsing of `<tool_call>` JSON
     - vLLM server tool extraction and schema validation
     - Round-trip of `AIMessage.tool_calls` and trace metadata
   - Integration tests:
     - `ReActMemAgent` on OpenAI backend with `tool_parser=openai`
     - `ReActMemAgent` on vLLM native backend with `tool_parser=hermes` and two-pass enabled
     - `WorkflowAgent` unchanged behavior

7) **Documentation**
   - README Providers section: describe backends, `tool_parser`, and two-pass behavior differences.
   - Native server usage with examples, including Hermes.
   - Known limitations and tips (e.g., if your OpenAI server lacks Hermes, enable client-side Hermes injection).


### Notes and trade-offs

- Keeping two-pass exclusively on the vLLM native path preserves strict control over private reasoning length without impacting the classical OpenAI tool-calling UX.
- Hermes support is provided in two ways: natively (preferred, when server supports it) or via client-side prompt injection and parsing.
- We avoid emulating the entire OpenAI Chat Completions spec in our native server; we expose a minimal, well-documented contract tailored to LangChain agents.



## OpenRouter GPT-OSS (Harmony) support

The `openai/gpt-oss-*` models use the Harmony format internally. When accessed through OpenRouter's OpenAI-compatible API, the provider handles Harmony rendering and tool-calling translation for you. Use the standard OpenAI backend in this framework:

- Add a provider to `config/config.yaml`:
  - `name`: e.g., `gpt_oss_20b_openrouter`
  - `model_name`: `openai/gpt-oss-20b`
  - `provider_backend`: `openai`
  - `tool_parser`: `openai`
  - `parsing_format`: `direct_response` (no `<think>` tags)
  - `api_config.base_url`: `https://openrouter.ai/api/v1`
  - `api_config.api_key_env`: `OPENROUTER_API_KEY` (ensure set in `.env`)

- Behavior notes:
  - Two-pass thinking is not used on this path; responses are single-pass via `ChatOpenAI`.
  - Chain-of-thought appears in Harmony `analysis` channel and is not surfaced in the final content; `LLMProvider.parse_response` should be `direct_response`.
  - Function/tool calling works via native OpenAI tool binding (`client.bind_tools(...)`). OpenRouter translates to Harmony under the hood and returns tool calls in the OpenAI-compatible schema.

- Example usage in an agent/run config: set the agent's provider name to `gpt_oss_20b_openrouter`.
