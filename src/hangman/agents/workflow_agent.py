# ./src/hangman/agents/workflow_agent.py
from __future__ import annotations

import json
import re
import yaml
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.prompts.workflow_agent import (
    MAIN_SYSTEM_PROMPT,
    UPDATE_MEMORY_SYSTEM_PROMPT,
    INITIAL_WORKING_MEMORY,
)

# --- Import memory tools (docstrings used in updater prompt) ---
from hangman.tools import (
    overwrite_memory,
    patch_memory,
    replace_in_memory,
    append_in_memory,
    delete_from_memory,
)


# =========================
# Agent State & Datamodels
# =========================

class AgentState(TypedDict):
    messages: List[BaseMessage]
    working_memory: str
    response: str
    thinking: str


StrategyName = Literal[
    "overwrite",
    "patch_and_replace",
    "append_and_delete",
]


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


# =========================
# Utility: Tool Executor
# =========================

class MemoryToolExecutor:
    """
    Thin wrapper around the existing tool functions. It injects current_memory and returns
    the updated string after each call. Supports both single-call and list-of-calls execution.
    """

    # Map canonical names -> imported tool callables
    TOOL_REGISTRY = {
        "overwrite_memory": overwrite_memory,
        "patch_memory": patch_memory,
        "replace_in_memory": replace_in_memory,
        "append_in_memory": append_in_memory,
        "delete_from_memory": delete_from_memory,
    }

    @classmethod
    def execute(
        cls,
        tool_calls: Union[ToolCall, List[ToolCall]],
        current_memory: str,
    ) -> str:
        if isinstance(tool_calls, ToolCall):
            tool_calls = [tool_calls]

        memory = current_memory
        for call in tool_calls:
            if call.name not in cls.TOOL_REGISTRY:
                raise ValueError(f"Unknown tool '{call.name}'. Allowed: {list(cls.TOOL_REGISTRY)}")
            tool_fn = cls.TOOL_REGISTRY[call.name]

            args = dict(call.arguments or {})
            # Inject current memory for write tools that accept it
            args["current_memory"] = memory

            result = tool_fn.invoke(args) if hasattr(tool_fn, "invoke") else tool_fn(**args)

            # Each tool may return:
            #  - str (new memory)
            #  - dict with {"new_memory": "..."}
            if isinstance(result, dict) and "new_memory" in result:
                memory = result["new_memory"]
            elif isinstance(result, str):
                memory = result
            else:
                raise RuntimeError(
                    f"Tool '{call.name}' returned unexpected shape: {type(result)}"
                )
        return memory


# =========================
# Workflow Agent
# =========================

class WorkflowAgent(BaseAgent):
    """
    A two-stage workflow agent:
      1) Responder LLM creates public `response` (+ optional private `thinking`)
      2) Updater LLM decides memory edits and returns tool-call JSON which is executed here

    Supported strategies for the updater step:
      - "overwrite"           -> overwrite_memory
      - "patch_and_replace"   -> patch_memory and/or replace_in_memory
      - "append_and_delete"   -> append_in_memory and/or delete_from_memory
    """

    def __init__(
        self,
        responder_llm_provider: LLMProvider,
        updater_llm_provider: LLMProvider,
        strategy: StrategyName,
        *,
        initial_memory: Optional[str] = None,
    ):
        super().__init__(llm_provider=responder_llm_provider)
        self.responder_llm = responder_llm_provider
        self.updater_llm = updater_llm_provider
        self.strategy: StrategyName = strategy
        self.turn_counter = 0
        self._initial_memory = initial_memory if initial_memory is not None else INITIAL_WORKING_MEMORY
        self.reset()

        # Precompute the updater prompt tool documentation text using tool docstrings.
        self._tool_guide_text = self._build_tool_guide_text(strategy)

        # Build graph
        self.workflow = self._build_workflow()

    # -------------
    # Graph wiring
    # -------------

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("update_memory", self._update_memory)

        workflow.set_entry_point("generate_response")
        workflow.add_edge("generate_response", "update_memory")
        workflow.add_edge("update_memory", END)

        return workflow.compile(checkpointer=MemorySaver())

    # --------------------
    # Workflow: Node logic
    # --------------------

    def _generate_response(self, state: AgentState) -> dict:
        """Responder LLM produces the public reply; no memory edits here."""
        print(f"---WORKFLOW AGENT: _generate_response---")
        prompt = MAIN_SYSTEM_PROMPT.format(working_memory=state["working_memory"])

        messages = [SystemMessage(content=prompt)] + state["messages"]
        result = self.responder_llm.invoke(messages, thinking=True)

        return {
            "thinking": result.get("thinking", ""),
            "response": result.get("response", result),
        }

    def _update_memory(self, state: AgentState) -> dict:
        """Updater LLM returns tool-call JSON; we execute it and persist the new memory.
        Retries up to 5 times; on failure, feeds the error back to the updater as context."""
        print(f"---WORKFLOW AGENT: _update_memory---")
        max_attempts = 5
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            print(f"---WORKFLOW AGENT: memory update attempt {attempt}/{max_attempts}---")

            # Build updater input; if we have a prior error, append it as guidance
            updater_input = self._build_updater_prompt(
                messages=state["messages"],
                working_memory=state["working_memory"],
                thinking=state["thinking"],
                response=state["response"],
                tool_guide_text=self._tool_guide_text,
            )
            if last_error:
                updater_input += (
                    "\n\n# Previous attempt failed\n"
                    "<last_error>\n"
                    f"{last_error}\n"
                    "</last_error>\n"
                    "Please correct the tool call(s) and return ONLY valid JSON per the required schema."
                )

            try:
                updater_messages = [HumanMessage(content=updater_input)]
                result = self.updater_llm.invoke(updater_messages)
            except Exception as e:
                last_error = f"Updater LLM invocation error: {e!r}"
                print(f"---WORKFLOW AGENT: {last_error}")
                continue

            try:
                tool_calls = self._parse_tool_calls(result.get("response", result))
                if not tool_calls:
                    print("---WORKFLOW AGENT: updater returned no tool calls; leaving memory unchanged.")
                    return {"working_memory": state["working_memory"]}

                new_memory = MemoryToolExecutor.execute(
                    tool_calls, current_memory=state["working_memory"]
                )
                print("---WORKFLOW AGENT: memory update succeeded.")
                return {"working_memory": new_memory}

            except Exception as e:
                last_error = f"Tool execution error: {e!r}"
                print(f"---WORKFLOW AGENT: {last_error}")
                # Loop to retry with the error fed back

        # If all attempts fail, keep memory unchanged
        print(f"---WORKFLOW AGENT: memory update failed after {max_attempts} attempts. Last error: {last_error}")
        return {"working_memory": state["working_memory"]}

    # -----------------
    # Public API
    # -----------------

    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        self.turn_counter += 1
        thread_config = {"configurable": {"thread_id": f"turn_{self.turn_counter}"}}

        # Fetch persistent state (main thread)
        main_thread = {"configurable": {"thread_id": "main_thread"}}
        try:
            current_state = self.workflow.get_state(config=main_thread)
            current_memory = current_state.values.get("working_memory", self._initial_memory)
            current_msgs = current_state.values.get("messages", [])
        except Exception:
            current_memory = self._initial_memory
            current_msgs = []

        initial = AgentState(
            messages=current_msgs + messages,
            working_memory=current_memory,
            response="",
            thinking="",
        )

        final_state = self.workflow.invoke(initial, config=thread_config)

        # Persist conversation + memory on main thread
        updated_messages = current_msgs + messages + [AIMessage(content=final_state["response"])]
        self.workflow.update_state(main_thread, {
            "messages": updated_messages,
            "working_memory": final_state["working_memory"],
        })

        return {
            "response": final_state["response"],
            "thinking": final_state["thinking"],
        }

    def get_state(self) -> AgentState:
        main_thread = {"configurable": {"thread_id": "main_thread"}}
        return self.workflow.get_state(config=main_thread).values

    def get_private_state(self) -> str:
        state = self.get_state()
        return state.get("working_memory", "")

    def reset(self) -> None:
        main_thread = {"configurable": {"thread_id": "main_thread"}}
        empty = AgentState(messages=[], working_memory=self._initial_memory, response="", thinking="")
        self.workflow.update_state(main_thread, empty)

    # -----------------
    # Updater Prompting
    # -----------------

    def _build_tool_guide_text(self, strategy: StrategyName) -> str:
        """
        Build a compact tool guide section sourced from each tool's docstring/description.
        We include exactly the tools allowed by the chosen strategy.
        """
        # LangChain @tool exposes description via .description (fallback to __doc__)
        def desc(tool_obj) -> str:
            return getattr(tool_obj, "description", None) or (tool_obj.__doc__ or "").strip()

        blocks: List[Tuple[str, str]] = []

        if strategy == "overwrite":
            blocks.append(("overwrite_memory", desc(overwrite_memory)))

        elif strategy == "patch_and_replace":
            blocks.append(("patch_memory", desc(patch_memory)))
            blocks.append(("replace_in_memory", desc(replace_in_memory)))

        elif strategy == "append_and_delete":
            blocks.append(("append_in_memory", desc(append_in_memory)))
            blocks.append(("delete_from_memory", desc(delete_from_memory)))

        out_lines = []
        out_lines.append("### Allowed tools for this update\n")
        out_lines.append("Return ONE of the following tool calls (or a SHORT list when necessary):\n")
        for name, d in blocks:
            out_lines.append(f"- **{name}**:\n")
            out_lines.append(self._indent_block(d, 2))
            out_lines.append("")
        out_lines.append(
            "Respond ONLY with JSON using one of these shapes:\n"
            "1) Single call: {\"name\": \"<tool_name>\", \"arguments\": { ... }}\n"
            "2) Multiple calls: [{\"name\": \"<tool_name>\", \"arguments\": { ... }}, ...]\n"
        )
        return "\n".join(out_lines)

    def _build_updater_prompt(
        self,
        *,
        messages: List[BaseMessage],
        working_memory: str,
        thinking: str,
        response: str,
        tool_guide_text: str,
    ) -> str:
        # Flatten recent dialogue (you may customize truncation outside)
        def fmt_msg(m: BaseMessage) -> str:
            role = "user" if isinstance(m, HumanMessage) else ("assistant" if isinstance(m, AIMessage) else m.type)
            return f"{role}: {m.content}"

        dialogue = "\n".join(fmt_msg(m) for m in messages)

        return UPDATE_MEMORY_SYSTEM_PROMPT.format(
            strategy=self.strategy,
            tool_guide=tool_guide_text,
            working_memory=working_memory,
            dialogue=dialogue,
            thinking=thinking,
            response=response,
        )

    # -----------------
    # Parsing helpers
    # -----------------

    @staticmethod
    def _parse_tool_calls(text: str) -> List[ToolCall]:
        """
        Robustly parse updater output. Accepts:
          - Pure JSON: {"name":"...", "arguments":{...}}
          - List of calls: [{"name":"...", "arguments":{...}}, ...]
          - JSON embedded in markdown code fences
        """
        if not text or not isinstance(text, str):
            return []

        # Extract JSON from code fences if present
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|$begin:math:display$.*?$end:math:display$)\s*```", text, flags=re.DOTALL)
        json_str = fence_match.group(1) if fence_match else text.strip()

        try:
            data = json.loads(json_str)
        except Exception:
            # Last resort: try to find the first {...} or [...]
            brute = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
            if not brute:
                return []
            try:
                data = json.loads(brute.group(1))
            except Exception:
                return []

        if isinstance(data, dict) and "name" in data:
            return [ToolCall(name=data["name"], arguments=data.get("arguments", {}))]
        if isinstance(data, list):
            calls: List[ToolCall] = []
            for item in data:
                if isinstance(item, dict) and "name" in item:
                    calls.append(ToolCall(name=item["name"], arguments=item.get("arguments", {})))
            return calls
        return []

    @staticmethod
    def _indent_block(text: str, spaces: int) -> str:
        pad = " " * spaces
        return "\n".join(pad + line for line in text.splitlines())


# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":

    CONFIG_PATH = "config.yaml"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    try:
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local")
        update_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local")
        print("✅ LLM Providers loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Providers: {e}")
        exit()

    # Change strategy here: "overwrite", "patch_and_replace", or "append_and_delete"
    agent = WorkflowAgent(responder_llm_provider=main_llm, updater_llm_provider=update_llm, strategy="append_and_delete")
    print(f"🤖 WorkflowAgent ({agent.strategy}) is ready. Type 'quit', 'exit', or 'q' to end.")

    messages = []
    while True:
        user_input = input("User > ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Ending session.")
            break

        messages.append(HumanMessage(content=user_input))

        output = agent.invoke(messages)

        messages.append(AIMessage(content=output["response"]))

        print("\n---ANSWER---")
        print(f"AI: {output['response']}")

        if "thinking" in output and output["thinking"]:
            print("\n---THINKING TRACE---")
            print(output["thinking"])

        print("\n---WORKING MEMORY---")
        print(agent.get_private_state())
        print("\n" + "="*50 + "\n")