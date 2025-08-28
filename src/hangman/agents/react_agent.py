import yaml
from typing import List, Any, Dict, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.tools import format_e2b_output_to_str
from hangman.prompts.react_agent import MAIN_SYSTEM_PROMPT


class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    thinking: str
    # Present only when code tool is enabled
    sandbox_files: List[str]


class ReActAgent(BaseAgent):
    """Stateless ReAct agent (no working memory). Accepts pre-initialized tools."""

    def __init__(self, main_llm_provider: LLMProvider, tools: Optional[List[Any]] = None):
        self.tools = tools or []  # no update_memory by default
        self.tools_by_name = {t.name: t for t in self.tools}
        self.model = main_llm_provider.client.bind_tools(self.tools)

        super().__init__(llm_provider=main_llm_provider)
        self.turn_counter = 0
        self.reset()

    def _build_workflow(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._call_model)
        graph.add_node("tools", self._tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._should_continue, {"continue": "tools", "end": END})
        graph.add_edge("tools", "agent")
        return graph.compile(checkpointer=MemorySaver())

    def _build_prompt_messages(self, state: AgentState) -> List[BaseMessage]:
        """Build messages for this pass: pruned history + full current-turn trace."""
        msgs: List[BaseMessage] = state.get("messages", [])
        last_human_idx = -1
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                last_human_idx = i
                break

        if last_human_idx == -1:
            history = [m for m in msgs if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))]
            current_turn: List[BaseMessage] = []
        else:
            history = msgs[:last_human_idx]
            history = [m for m in history if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))]
            current_turn = msgs[last_human_idx:]

        return history + current_turn

    def _should_continue(self, state: AgentState) -> str:
        if not state["messages"][-1].tool_calls:
            return "end"
        return "continue"

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        # Stateless prompt (no working_memory)
        system_prompt = SystemMessage(content=MAIN_SYSTEM_PROMPT)
        prompt_messages = [system_prompt] + self._build_prompt_messages(state)

        response_obj = self.model.invoke(prompt_messages)
        parsed = self.llm_provider.parse_response(response_obj.content or "")
        has_tool_calls = bool(getattr(response_obj, "tool_calls", None))
        if not has_tool_calls:
            response_obj.content = parsed["response"]
        # Attach thinking so we can reconstruct within-turn chronology
        try:
            response_obj.additional_kwargs = getattr(response_obj, "additional_kwargs", {}) or {}
            response_obj.additional_kwargs["thinking"] = parsed.get("thinking", "")
        except Exception:
            pass
        return {"messages": state["messages"] + [response_obj], "thinking": parsed.get("thinking", "")}

    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        tool_call = state["messages"][-1].tool_calls[0]
        tool = self.tools_by_name[tool_call["name"]]
        tool_args = dict(tool_call["args"])

        if tool.name == "web_search":
            result = tool.invoke(tool_args)
            tool_message = ToolMessage(content=result, name=tool_call["name"], tool_call_id=tool_call["id"])
            return {"messages": state["messages"] + [tool_message]}

        if tool.name == "code_interpreter":
            observation = tool.invoke(tool_args)
            content_str = format_e2b_output_to_str(observation)
            tool_message = ToolMessage(content=content_str, name=tool_call["name"], tool_call_id=tool_call["id"])
            return {"messages": state["messages"] + [tool_message], "sandbox_files": observation.get("files", [])}

        # Fallback
        result = tool.invoke(tool_args)
        tool_message = ToolMessage(content=str(result), name=tool_call["name"], tool_call_id=tool_call["id"])
        return {"messages": state["messages"] + [tool_message]}

    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        self.turn_counter += 1
        turn_cfg = {"configurable": {"thread_id": f"react_stateless_turn_{self.turn_counter}"}}
        main_cfg = {"configurable": {"thread_id": "main_thread"}}

        initial_state: Dict[str, Any] = {"messages": messages}
        # If a code tool is present, carry sandbox_files across turns
        if "code_interpreter" in self.tools_by_name:
            try:
                current = self.get_state()
                initial_state["sandbox_files"] = current.get("sandbox_files", [])
            except Exception:
                initial_state["sandbox_files"] = []

        final_state = self.workflow.invoke(initial_state, config=turn_cfg)

        # Persist pruned conversation: keep prior + only final AI (drop tool calls and ToolMessages)
        full_msgs: List[BaseMessage] = final_state["messages"]
        base_len = len(messages)
        prior_msgs = full_msgs[:base_len]
        turn_msgs = full_msgs[base_len:]

        pruned_turn_msgs: List[BaseMessage] = []
        for m in reversed(turn_msgs):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                pruned_turn_msgs = [m]
                break
        persisted_messages = prior_msgs + pruned_turn_msgs

        update_payload: Dict[str, Any] = {"messages": persisted_messages}
        if "code_interpreter" in self.tools_by_name:
            update_payload["sandbox_files"] = final_state.get("sandbox_files", [])
        self.workflow.update_state(main_cfg, update_payload)

        # Final response content
        final_response = ""
        for msg in reversed(full_msgs):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_response = msg.content
                break

        # Build current-turn thinking/tool trace
        trace_lines: List[str] = []
        for m in turn_msgs:
            if isinstance(m, AIMessage):
                thought = getattr(m, "additional_kwargs", {}).get("thinking", "")
                if thought:
                    trace_lines.append("---Explicit Thought---")
                    trace_lines.append(thought)
                if m.content:
                    trace_lines.append(str(m.content))
                trace_lines.append(m.pretty_repr())
            elif isinstance(m, ToolMessage):
                trace_lines.append(m.pretty_repr())
        full_thinking = "\n".join(trace_lines)

        return {"response": final_response, "thinking": full_thinking}

    def get_state(self) -> Dict[str, Any]:
        snap = self.workflow.get_state({"configurable": {"thread_id": "main_thread"}})
        return snap.values if snap else {}

    def get_private_state(self) -> str:
        # Stateless: return empty string for private state
        return ""

    def reset(self) -> None:
        empty: AgentState = AgentState(messages=[], thinking="")
        if "code_interpreter" in self.tools_by_name:
            empty["sandbox_files"] = []
        self.workflow.update_state({"configurable": {"thread_id": "main_thread"}}, empty)

    def close(self) -> None:
        return


if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    with open(CONFIG_PATH, "r") as f:
        yaml.safe_load(f)

    main_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local_vllm_native")
    agent = ReActAgent(main_llm_provider=main_llm)
    print("ReActAgent (stateless) ready.")
    messages: List[BaseMessage] = []
    while True:
        user = input("User > ")
        if user.lower() in ["q", "quit", "exit"]:
            break
        messages.append(HumanMessage(content=user))
        output = agent.invoke(messages)
        messages.append(AIMessage(content=output["response"]))
        print("\n---ANSWER---")
        print(output["response"])
        if "thinking" in output and output["thinking"]:
            print("\n---THINKING TRACE---")
            print(output["thinking"])
        print("\n" + "="*50 + "\n")
        
