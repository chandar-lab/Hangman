from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests
from langchain_core.messages import AIMessage, BaseMessage

from .base_client import BaseClient, ToolSchema, ToolChoice


class ChatVllm(BaseClient):
    """
    LangChain-compatible chat client for the native vLLM server.

    Behavior goals:
    - Return an AIMessage with `.content` (raw text) and optional `.tool_calls`
    - Do NOT split response/thinking; the agent will parse `.content` via
      LLMProvider.parse_response(...)
    - If server returns tool calls, convert them to the structure expected by
      ReActMemAgent: [{"id", "name", "args"}]
    """

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        tool_parser: str = "hermes",
        two_pass: bool = True,
        think_tag: str = "think",
        max_thinking_tokens: int = 256,
        max_response_tokens: int = 1024,
        request_timeout_s: int = 120,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_parser=tool_parser,  # "openai" | "hermes"; default Hermes for Qwen
        )
        self.two_pass = bool(two_pass)
        self.think_tag = str(think_tag or "think")
        self.max_thinking_tokens = int(max_thinking_tokens)
        self.max_response_tokens = int(max_response_tokens)
        self.request_timeout_s = int(request_timeout_s)

    def _build_payload(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": self.to_openai_chat(messages),
            "temperature": self.temperature,
            "two_pass": self.two_pass,
            "think_tag": self.think_tag,
            "max_thinking_tokens": self.max_thinking_tokens,
            "max_response_tokens": self.max_response_tokens,
            "tool_parser": self.tool_parser,
        }

        # Attach tool specs if any are bound
        if getattr(self, "_bound_tool_specs", None):
            payload["tools"] = [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                }
                for spec in self._bound_tool_specs
            ]
            payload["tool_choice"] = self._tool_choice  # "auto" | "none" | {"name": ...}

        return payload

    def bind_tools(self, tools: List[Any], tool_choice: ToolChoice = "auto") -> "ChatVllm":
        """Expose bind_tools to match ChatOpenAI API; store schemas and choice then return self."""
        super().bind_tools(tools, tool_choice=tool_choice)
        return self

    def _compose_content(
        self,
        thinking: str,
        pre_tool_text: Optional[str],
        response: Optional[str],
        server_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Compose raw assistant text for AIMessage.content.
        - Always include a closed think block if `thinking` is present
        - For tool calls: prefer pre_tool_text; otherwise fallback to response
        - For normal replies: use response
        """
        parts: List[str] = []
        thinking_text = (thinking or "").strip()
        if thinking_text:
            parts.append(f"<{self.think_tag}>{thinking_text}</{self.think_tag}>")

        # Prefer pre_tool_text when tools are present; caller will pass the right pair
        chosen_text = pre_tool_text if (pre_tool_text is not None) else response
        if chosen_text:
            parts.append(str(chosen_text).strip())

        # Reconstruct a textual <tool_call> block to preserve trace visibility
        if server_tool_calls:
            # Build JSON payload as single object or array depending on count
            calls_payload: Any
            if len(server_tool_calls) == 1:
                one = server_tool_calls[0]
                calls_payload = {
                    "name": one.get("name"),
                    "arguments": one.get("arguments", {}),
                }
            else:
                calls_payload = [
                    {"name": c.get("name"), "arguments": c.get("arguments", {})}
                    for c in server_tool_calls
                ]
            try:
                calls_json = json.dumps(calls_payload, ensure_ascii=False)
                parts.append("<tool_call>")
                parts.append(calls_json)
                parts.append("</tool_call>")
            except Exception:
                # If serialization fails, skip embedding textual tool_call
                pass

        return "\n".join(p for p in parts if p)

    def _to_langchain_tool_calls(self, server_calls: Any) -> List[Dict[str, Any]]:
        """Map server `tool_calls` to ReAct-compatible structure: {id, name, args}."""
        out: List[Dict[str, Any]] = []
        if not isinstance(server_calls, list):
            return out
        for idx, call in enumerate(server_calls):
            try:
                name = call.get("name")
                arguments = call.get("arguments", {})
                call_id = call.get("id") or f"call_{idx + 1}"
                out.append({"id": call_id, "name": str(name), "args": dict(arguments or {})})
            except Exception:
                continue
        return out

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        url = (self.base_url or "").rstrip("/") + "/generate"
        payload = self._build_payload(messages)

        try:
            resp = requests.post(url, json=payload, timeout=self.request_timeout_s)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return AIMessage(content=f"Error: {e}", tool_calls=[])

        # Server schema: {response: str, thinking: str, tool_calls?: [...], pre_tool_text?: str}
        thinking = data.get("thinking", "")
        response_text = data.get("response", "")
        server_tool_calls = data.get("tool_calls")
        pre_tool_text = data.get("pre_tool_text")

        # When tool calls are present, content includes think + pre_tool_text; otherwise think + response
        has_tools = isinstance(server_tool_calls, list) and len(server_tool_calls) > 0
        content = self._compose_content(
            thinking=thinking,
            pre_tool_text=pre_tool_text if has_tools else None,
            response=response_text if not has_tools else None,
            server_tool_calls=server_tool_calls if has_tools else None,
        )
        tool_calls = self._to_langchain_tool_calls(server_tool_calls)

        return AIMessage(content=content, tool_calls=tool_calls)


