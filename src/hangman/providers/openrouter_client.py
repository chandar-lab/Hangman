from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI
from langchain_core.messages import AIMessage, BaseMessage

from .base_client import BaseClient, ToolSchema, ToolChoice


class ChatOpenRouter(BaseClient):
    """
    LangChain-compatible chat client using OpenRouter's /chat/completions
    via the OpenAI SDK.

    Goals:
    - Keep .bind_tools(...) on the CLIENT (not provider)
    - Return an AIMessage with:
        * .content containing <think>...</think> + final content (if reasoning is present)
        * .tool_calls as [{"id","name","args"}] for your ReAct flow
        * .additional_kwargs["reasoning"] carrying raw analysis text
    - Don't rely on OpenAI Responses API (to avoid the routing mismatch).
    """

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        request_timeout_s: int = 120,
        include_reasoning: bool = True,     # OpenRouter often returns reasoning w/o this, but keep knob
        reasoning_effort: Optional[str] = None,  # 'low'|'medium'|'high'|'auto' or None
        think_tag: str = "think",           # for <think> ... </think> wrapping of analysis
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_parser="openai",  # tools are standard OpenAI format here
        )
        self._sdk = OpenAI(base_url=base_url, api_key=api_key)
        self.request_timeout_s = int(request_timeout_s)
        self.include_reasoning = bool(include_reasoning)
        self.reasoning_effort = reasoning_effort
        self.think_tag = (think_tag or "think").strip("<>/ ")

        # populated by bind_tools(...)
        self._bound_tool_specs: Optional[List[ToolSchema]] = None
        self._tool_choice: ToolChoice = "auto"

    # ---------- public API ----------

    def bind_tools(self, tools: List[Any], tool_choice: ToolChoice = "auto") -> "ChatOpenRouter":
        """Client-level binding (keeps compatibility with your repo)."""
        super().bind_tools(tools, tool_choice=tool_choice)  # stores _bound_tool_specs / _tool_choice
        return self

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        wire_messages = self.to_openai_chat(messages)

        # Reasoning knob (provider-specific); OpenRouter often works w/out it, but support both
        extra_body: Dict[str, Any] = {}
        if self.include_reasoning or self.reasoning_effort:
            rb: Dict[str, Any] = {}
            if self.include_reasoning:
                rb["include"] = True
            if self.reasoning_effort and self.reasoning_effort.lower() != "auto":
                rb["effort"] = self.reasoning_effort.lower()
            if rb:
                extra_body["reasoning"] = rb

        tools_payload = None
        if self._bound_tool_specs:
            tools_payload = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
                for spec in self._bound_tool_specs
            ]

        try:
            resp = self._sdk.chat.completions.create(
                model=self.model_name,
                messages=wire_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools_payload,
                tool_choice=self._tool_choice,
                extra_body=extra_body or None,
                timeout=self.request_timeout_s,
            )
        except Exception as e:
            return AIMessage(content=f"Error: {e}", tool_calls=[])

        raw = resp.model_dump()
        choice = raw["choices"][0]
        msg = choice["message"]

        final_text = msg.get("content") or ""
        reasoning_text = msg.get("reasoning") or ""  # OpenRouter returns a string when exposed

        # Make it compatible with your parse_response(...): embed <think>...</think> at the top.
        content = self._compose_content_with_think(final_text, reasoning_text)

        tool_calls = self._to_langchain_tool_calls(msg.get("tool_calls"))

        # Attach useful metadata
        addl = {
            "reasoning": reasoning_text,                   # raw analysis (also embedded in content)
            "usage": raw.get("usage"),
            "finish_reason": choice.get("finish_reason"),
            "provider": "openrouter",
            "model": raw.get("model"),
        }

        # print(f"--- From OpenRouter Client ---")
        # print(f"Content: {content}")
        # print(f"Reasoning: {reasoning_text}")
        # print(f"--- End OpenRouter Client ---")

        return AIMessage(content=content, tool_calls=tool_calls, additional_kwargs=addl)

    # ---------- helpers ----------

    def _compose_content_with_think(self, final_text: str, reasoning_text: str) -> str:
        reasoning_text = (reasoning_text or "").strip()
        if reasoning_text:
            return f"<{self.think_tag}>{reasoning_text}</{self.think_tag}>\n{final_text.strip() if final_text else ''}".strip()
        return (final_text or "").strip()

    def _to_langchain_tool_calls(self, server_calls: Any) -> List[Dict[str, Any]]:
        """
        Map OpenAI tool_calls to your ReAct-compatible structure: {id, name, args}.
        """
        out: List[Dict[str, Any]] = []
        if not isinstance(server_calls, list):
            return out

        for idx, tc in enumerate(server_calls):
            if tc.get("type") != "function":
                continue
            fn = tc.get("function", {}) or {}
            name = fn.get("name")
            args_raw = fn.get("arguments", "{}")

            # OpenAI returns arguments as a JSON string
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except Exception:
                args = {}

            call_id = tc.get("id") or f"call_{idx+1}"
            out.append({"id": call_id, "name": str(name), "args": dict(args)})

        return out