from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage


ToolChoice = Union[Literal["auto", "none"], Dict[str, str]]


@dataclass
class ToolSchema:
    """
    Minimal schema describing a tool for prompt exposition and server calls.

    - name: canonical tool name
    - description: concise natural language description
    - parameters: JSONSchema-like dict describing arguments
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseClient(ABC):
    """
    Abstract base for chat clients used by agents.

    Implementations must emulate the LangChain Chat model interface expected by
    ReAct-style agents:
      - bind_tools(tools, tool_choice) -> client-like object
      - invoke(messages) -> AIMessage with `.content` and optional `.tool_calls`

    This class also provides shared helpers for message normalization and
    extracting tool schemas from common LangChain tool objects.
    """

    def __init__(
        self,
        *,
        model_name: str,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        tool_parser: Literal["openai", "hermes"] = "openai",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_parser = tool_parser

        self._bound_tools: List[Any] = []
        self._bound_tool_specs: List[ToolSchema] = []
        self._tool_choice: ToolChoice = "auto"

    def bind_tools(self, tools: List[Any], tool_choice: ToolChoice = "auto") -> "BaseClient":
        """
        Record tools and tool choice, returning a client-like object.

        Subclasses may override to return a distinct bound instance. The default
        behavior mutates and returns self, which is sufficient for custom
        clients like ChatVllm.
        """
        self._bound_tools = list(tools or [])
        self._bound_tool_specs = self.build_tool_specs(self._bound_tools)
        self._tool_choice = tool_choice
        return self

    @abstractmethod
    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Execute a chat completion and return a LangChain AIMessage.

        Requirements:
          - AIMessage.content: raw assistant text
          - AIMessage.tool_calls (optional): list of dicts with keys
            {"id": str, "name": str, "args": dict}
        """
        raise NotImplementedError

    # --------------------------
    # Shared helper utilities
    # --------------------------
    @staticmethod
    def to_openai_chat(messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        Convert LangChain messages to OpenAI-style chat dicts: {role, content}.
        """
        out: List[Dict[str, str]] = []
        for m in messages:
            role = getattr(m, "type", None) or "user"
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            out.append({"role": role, "content": str(m.content)})
        return out

    @staticmethod
    def build_tool_specs(tools: List[Any]) -> List[ToolSchema]:
        """
        Best-effort extraction of name, description, and JSONSchema parameters
        from LangChain tool objects.
        """
        specs: List[ToolSchema] = []
        for t in tools:
            name = getattr(t, "name", None) or getattr(t, "__name__", "tool")
            description = getattr(t, "description", None) or (getattr(t, "__doc__", "") or "").strip()

            schema: Dict[str, Any] = {}

            args_schema = getattr(t, "args_schema", None)
            if args_schema is not None:
                try:
                    if hasattr(args_schema, "model_json_schema"):
                        schema = args_schema.model_json_schema()  # Pydantic v2
                    elif hasattr(args_schema, "schema"):
                        schema = args_schema.schema()  # Pydantic v1
                except Exception:
                    schema = {}

            if not schema:
                # Fallback: attempt to read a dict-ish signature
                raw_params = getattr(t, "args", None)
                if isinstance(raw_params, dict):
                    schema = {
                        "type": "object",
                        "properties": {k: {"type": "string"} for k in raw_params.keys()},
                        "required": list(raw_params.keys()),
                    }
                else:
                    schema = {"type": "object", "properties": {}}

            specs.append(ToolSchema(name=str(name), description=str(description or name), parameters=schema))

        return specs


