import os
import yaml
import logging
from typing import List, Any, Literal, Optional, Tuple
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
import requests

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Type Definitions ---
class ModelOutput(TypedDict):
    response: str
    thinking: str

class APIConfig(TypedDict):
    base_url: str
    api_key_env: str

class GenerationConfig(TypedDict):
    temperature: float
    max_tokens: int
    # Optional, used when thinking=True and/or two-pass logic is enabled
    two_pass: Optional[bool]
    think_tag: Optional[Literal["think", "thinking"]]
    max_thinking_tokens: Optional[int]
    max_response_tokens: Optional[int]

class ProviderConfig(TypedDict):
    name: str
    model_name: str
    parsing_format: Optional[Literal["think_tags", "direct_response"]]
    # Backend selection: "openai" (ChatOpenAI over HTTP) or "vllm_native" (Python API)
    provider_backend: Optional[Literal["openai", "vllm_native"]]
    api_config: APIConfig
    generation_config: GenerationConfig

# --- LLM Provider ---
class LLMProvider:
    """
    A unified wrapper for LLM providers. It uses a configuration-driven approach
    to handle different models and output parsing formats.
    """
    def __init__(self, config: ProviderConfig):
        self.config = config
        # Determine backend; default to OpenAI-compatible HTTP client
        self.backend: str = self.config.get("provider_backend", "openai")  # type: ignore[assignment]
        # Holders for backend-specific clients
        self.client: Any = None
        self.vllm: Any = None
        self.vllm_tokenizer: Any = None
        self.client = self._create_client()

    def _create_client(self) -> Any:
        """Create and initialize the underlying model client based on backend."""
        if self.backend == "vllm_native":
            # External HTTP server mode; no local model initialization here
            logging.info(
                "Using vLLM native HTTP backend at %s",
                self.config["api_config"].get("base_url"),
            )
            return None

        # Default: OpenAI-compatible HTTP client (vLLM server, OpenRouter, etc.)
        api_key_env = self.config["api_config"].get("api_key_env")
        base_url = self.config["api_config"].get("base_url")
        api_key = os.environ.get(api_key_env) if api_key_env else None

        # Allow missing API key for localhost endpoints (common for local vLLM servers)
        if api_key_env and api_key is None:
            if isinstance(base_url, str) and ("localhost" in base_url or "127.0.0.1" in base_url):
                logging.warning(
                    "API key env '%s' not set, but base_url looks local (%s). Proceeding without a key.",
                    api_key_env,
                    base_url,
                )
                api_key = ""
            else:
                raise ValueError(f"API key environment variable '{api_key_env}' not set.")

        return ChatOpenAI(
            model=self.config["model_name"],
            base_url=base_url,
            api_key=api_key or "",  # Some OpenAI-compatible servers accept empty tokens for local use
            temperature=self.config["generation_config"].get("temperature", 0.7),
            max_tokens=self.config["generation_config"].get("max_tokens", 4096),
        )

    def _parse_with_think_tags(self, text: str) -> ModelOutput:
        """Parses text containing <think>...</think> tags."""
        if "<think>" in text:
            tag_word = "think"
        elif "<thinking>" in text:
            tag_word = "thinking"
        else:
            logging.warning(f"No thinking tags found in the response. Treating as direct response. The response is:\n--\n{text}\n--")
            return {"response": text, "thinking": ""}
        think_start_tag = f"<{tag_word}>"
        think_end_tag = f"</{tag_word}>"
        start_index = text.find(think_start_tag)
        end_index = text.find(think_end_tag)

        if start_index != -1 and end_index > start_index:
            thinking = text[start_index + len(think_start_tag):end_index].strip()
            response = text[end_index + len(think_end_tag):].strip()
            return {"response": response, "thinking": thinking}

        # If we saw an opening tag but no closing tag, treat the remainder as thinking only
        # and synthesize a closing tag. Public response becomes a default error message.
        if start_index != -1 and end_index == -1:
            thinking_partial = text[start_index + len(think_start_tag):].strip()
            thinking_full = f"{thinking_partial}{think_end_tag}"
            return {
                "response": "Error: model did not close the thinking segment; no public answer generated.",
                "thinking": thinking_full,
            }

        # Fallback: treat as direct response
        return {"response": text, "thinking": ""}

    def _parse_direct_response(self, text: str) -> ModelOutput:
        """Handles models that give a direct response with no thinking."""
        return {"response": text, "thinking": ""}
    
    def parse_response(self, full_response_text: str) -> ModelOutput:
        """
        NEW: Public method to parse a raw text response into a ModelOutput,
        extracting <think> tags based on the provider's configuration.
        """
        parsing_format = self.config.get("parsing_format", "direct_response")
        if parsing_format == "think_tags":
            return self._parse_with_think_tags(full_response_text)
        elif parsing_format == "direct_response":
            return self._parse_direct_response(full_response_text)
        else:
            logging.warning(f"Unsupported parsing format '{parsing_format}'. Defaulting to direct response.")
            return self._parse_direct_response(full_response_text)

    def invoke(
        self,
        messages: List[BaseMessage],
        thinking: bool = False,
        *,
        max_thinking_tokens: Optional[int] = None,
        max_response_tokens: Optional[int] = None,
        think_tag: Optional[Literal["think", "thinking"]] = None,
        two_pass: Optional[bool] = None,
    ) -> ModelOutput:
        """
        Invoke the model. If thinking=True and two-pass is enabled, perform a two-pass
        private-thought then public-answer generation, preserving the output shape.
        """
        gen_cfg = self.config.get("generation_config", {})
        # Determine settings and whether to run two-pass
        cfg_two_pass = gen_cfg.get("two_pass", False)
        two_pass_enabled = (two_pass if two_pass is not None else cfg_two_pass) and thinking
        effective_think_tag = think_tag or gen_cfg.get("think_tag", "think")
        thinking_cap = max_thinking_tokens if max_thinking_tokens is not None else gen_cfg.get("max_thinking_tokens", 256)
        response_cap = max_response_tokens if max_response_tokens is not None else gen_cfg.get("max_response_tokens", gen_cfg.get("max_tokens", 1024))

        # Always route vLLM native backend through the HTTP server, regardless of two-pass
        if self.backend == "vllm_native":
            return self._invoke_vllm_http(
                messages,
                effective_think_tag,
                thinking_cap,
                response_cap,
                two_pass=two_pass_enabled,
            )

        # OpenAI-compatible backend
        if two_pass_enabled:
            return self._invoke_openai_two_pass(messages, effective_think_tag, thinking_cap, response_cap)

        # Legacy single-pass behavior
        try:
            response_obj = self.client.invoke(messages)
            full_response_text = response_obj.content
        except Exception as e:
            logging.error(f"An error occurred while calling the model server: {e}")
            return {"response": "Error: Could not connect to the model server.", "thinking": ""}

        parsed_output = self.parse_response(full_response_text)
        if not thinking:
            # If parser detected an unterminated think tag, keep the synthesized thinking trace
            # and surface a default error in the public response as per policy.
            if parsed_output.get("thinking") and parsed_output.get("response", "").startswith("Error: model did not close the thinking"):
                return parsed_output
            parsed_output["thinking"] = ""
        return parsed_output

    # --- Backend-specific two-pass implementations ---
    def _invoke_openai_two_pass(
        self,
        messages: List[BaseMessage],
        tag: Literal["think", "thinking"],
        max_thinking_tokens: int,
        max_response_tokens: int,
    ) -> ModelOutput:
        thinking_core, response_prefix, had_tag, detected_tag = self._first_pass_openai(messages, tag, max_thinking_tokens)
        if not had_tag:
            # No-think fallback already returned entire text as response via response_prefix
            return {"response": response_prefix, "thinking": ""}

        # Use the detected tag if present, otherwise fall back to requested tag
        used_tag = detected_tag or tag
        closed_think_block = f"<{used_tag}>{thinking_core}</{used_tag}>"
        second_text = self._second_pass_openai(messages, closed_think_block, max_response_tokens)

        combined = (response_prefix + (" " if response_prefix and second_text else "") + second_text).strip()
        return {"response": combined, "thinking": thinking_core}

    def _first_pass_openai(
        self,
        messages: List[BaseMessage],
        tag: Literal["think", "thinking"],
        max_thinking_tokens: int,
    ) -> Tuple[str, str, bool, Optional[str]]:
        """Run first pass to elicit <tag>...</tag> with a cap and stop.
        Returns (thinking_core, response_prefix, had_tag, detected_tag).
        If no opening tag is found, returns ("", full_text, False).
        """
        # Be robust: stop on either closing tag
        stop = ["</think>", "</thinking>"]
        client = ChatOpenAI(
            model=self.config["model_name"],
            base_url=self.config["api_config"].get("base_url"),
            api_key=os.environ.get(self.config["api_config"].get("api_key_env", ""), ""),
            temperature=self.config["generation_config"].get("temperature", 0.7),
            max_tokens=max_thinking_tokens,
        )
        try:
            obj = client.invoke(messages, stop=stop)
            text = obj.content
        except Exception as e:
            logging.error(f"OpenAI two-pass first call failed: {e}")
            return "", "Error: model call failed.", False, None

        # Detect which tag the model used (prefer complete pairs)
        candidates = [
            ("think", "<think>", "</think>"),
            ("thinking", "<thinking>", "</thinking>"),
        ]

        best = None  # tuple: (tag_name, start_idx, end_idx)
        for name, open_t, close_t in candidates:
            s = text.find(open_t)
            e = text.find(close_t)
            if s != -1 and e > s:
                best = (name, s, e)
                break

        if best is not None:
            name, s, e = best
            open_t = f"<{name}>"
            close_t = f"</{name}>"
            thinking_core = text[s + len(open_t): e].strip()
            response_prefix = text[e + len(close_t):].strip()
            return thinking_core, response_prefix, True, name

        # If no complete pair, check for any opening tag without closing
        for name, open_t, _ in candidates:
            s = text.find(open_t)
            if s != -1:
                thinking_core = text[s + len(open_t):].strip()
                return thinking_core, "", True, name

        # No tags found at all
        return "", text.strip(), False, None

    def _second_pass_openai(
        self,
        messages: List[BaseMessage],
        closed_think_block: str,
        max_response_tokens: int,
    ) -> str:
        """Continue generation after the closed think block by appending an internal AIMessage."""
        second_messages = messages + [AIMessage(content=closed_think_block)]
        client = ChatOpenAI(
            model=self.config["model_name"],
            base_url=self.config["api_config"].get("base_url"),
            api_key=os.environ.get(self.config["api_config"].get("api_key_env", ""), ""),
            temperature=self.config["generation_config"].get("temperature", 0.7),
            max_tokens=max_response_tokens,
        )
        try:
            obj = client.invoke(second_messages)
            return obj.content.strip()
        except Exception as e:
            logging.error(f"OpenAI two-pass second call failed: {e}")
            return ""

    def _invoke_vllm_http(
        self,
        messages: List[BaseMessage],
        tag: Literal["think", "thinking"],
        max_thinking_tokens: int,
        max_response_tokens: int,
        *,
        two_pass: bool,
    ) -> ModelOutput:
        base_url = self.config["api_config"].get("base_url")
        if not base_url:
            return {"response": "Error: vLLM native base_url not configured.", "thinking": ""}

        def _to_chat(messages: List[BaseMessage]):
            out = []
            for m in messages:
                role = getattr(m, "type", None) or "user"
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                out.append({"role": role, "content": m.content})
            return out

        payload = {
            "messages": _to_chat(messages),
            "temperature": self.config["generation_config"].get("temperature", 0.3),
            "two_pass": bool(two_pass),
            "think_tag": tag,
            "max_thinking_tokens": int(max_thinking_tokens),
            "max_response_tokens": int(max_response_tokens),
        }

        try:
            resp = requests.post(base_url.rstrip("/") + "/generate", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "response" in data and "thinking" in data:
                return {"response": str(data["response"]), "thinking": str(data["thinking"])}
            return {"response": "Error: invalid server response.", "thinking": ""}
        except Exception as e:
            logging.error("vLLM native HTTP call failed: %s", e)
            return {"response": f"Error: {e}", "thinking": ""}


# --- Helper/Factory Function ---
def load_llm_provider(config_path: str, provider_name: str) -> LLMProvider:
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    provider_configs = full_config.get("providers", [])
    target_config = next((p for p in provider_configs if p["name"] == provider_name), None)
    
    if not target_config:
        raise ValueError(f"Provider '{provider_name}' not found in {config_path}")
        
    return LLMProvider(target_config)