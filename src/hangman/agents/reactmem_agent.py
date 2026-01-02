import os
import yaml
import json
from typing import List, Any, Dict, Optional
import warnings

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
# from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# --- Project-Specific Imports ---
from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider

from hangman.tools import format_e2b_output_to_str, patch_memory, replace_in_memory, delete_from_memory, append_in_memory, overwrite_memory
from hangman.prompts.reactmem_agent import MAIN_SYSTEM_PROMPT, INITIAL_WORKING_MEMORY, SAVE_SECRET_HINT

# --- Agent State and Class Definition ---

class AgentState(TypedDict, total=False):
    """The state for the memoryful ReAct agent."""
    messages: List[BaseMessage]
    thinking: str
    working_memory: str
    # Present only when code tool is enabled
    sandbox_files: List[str]

class ReActMemAgent(BaseAgent):
    """
    An agent that uses the ReAct (Reason+Act) paradigm.
    Its only tool is the ability to edit its own working memory.
    """
    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Optional[List[Any]] = None,
        strategy: str = "overwrite",
        add_save_secret_hint: bool = True,
    ):
        """Memoryful ReAct agent that accepts a pre-initialized tool list.

        If tools is None, defaults to [patch_memory, replace_in_memory].
        """
        if tools is None and strategy is None:
            raise ValueError("Either tools or strategy must be provided.")
        if tools is not None and strategy is not None:
            warnings.warn("Both tools and strategy are provided. The strategy will be ignored. Using tools.")
        if tools:
            self.tools = tools
        else:
            if strategy == "overwrite":
                self.tools = [overwrite_memory]
            elif strategy == "patch_and_replace":
                self.tools = [patch_memory, replace_in_memory]
            elif strategy == "append_and_delete":
                self.tools = [delete_from_memory, append_in_memory]
            else:
                raise ValueError("Invalid strategy. Must be one of: overwrite, patch_and_replace, append_and_delete.")

        self.tools_by_name = {t.name: t for t in self.tools}
        self._has_code_tool = "code_interpreter" in self.tools_by_name

        # Bind the tools to the LLM
        self.model = llm_provider.client.bind_tools(self.tools)

        super().__init__(llm_provider=llm_provider)
        self.turn_counter = 0  # Used for thread management
        self.reset()

        # Save system prompt as attribute
        self.main_system_prompt = MAIN_SYSTEM_PROMPT
        if add_save_secret_hint:
            self.main_system_prompt += SAVE_SECRET_HINT

    def _build_workflow(self) -> StateGraph:
        """Constructs the agent's LangGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._tool_node)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=MemorySaver())

    # --- Graph Nodes and Logic ---

    def _build_prompt_messages(self, state: AgentState) -> List[BaseMessage]:
        """Build the messages to send to the LLM for this pass.

        Rules:
        - History must include only Human messages and final AI messages (no tool-calls).
        - The current-turn segment (from the last HumanMessage onward) must include the full
          within-turn trace: Human, AI(tool_call), ToolMessage(s), etc.
        """
        msgs: List[BaseMessage] = state.get("messages", [])
        # Find last HumanMessage index
        last_human_idx = -1
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                last_human_idx = i
                break

        if last_human_idx == -1:
            # No human yet: treat everything as history and filter to final AI+humans only
            history = [m for m in msgs if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))]
            current_turn: List[BaseMessage] = []
        else:
            history = msgs[:last_human_idx]
            # Keep only Human and final AI in history
            history = [m for m in history if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))]
            # Current turn includes the last human + any AI(tool_call) and ToolMessages
            current_turn = msgs[last_human_idx:]

        return history + current_turn

    def _should_continue(self, state: AgentState):
        """Determines whether to continue with a tool call or end."""
        if not state["messages"][-1].tool_calls:
            return "end"
        return "continue"

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """Invokes the LLM with the current state to decide on an action or response."""
        print("---NODE: AGENT---")
        # Format the system prompt with the current working memory
        system_prompt = SystemMessage(content=self.main_system_prompt.format(working_memory=state.get("working_memory", "")))
        # Build prompt: system + pruned history + full current-turn trace
        prompt_messages = [system_prompt] + self._build_prompt_messages(state)

        response_obj = self.model.invoke(prompt_messages)
        parsed_output = self.llm_provider.parse_response(response_obj.content or "")
        # If the model is invoking a tool, keep the original content so the
        # pre-tool reasoning remains visible in the next pass. Clean only on final.
        has_tool_calls = bool(getattr(response_obj, "tool_calls", None))
        if not has_tool_calls:
            response_obj.content = parsed_output["response"]
        # Attach per-pass thinking to the AI message so we can reconstruct within-turn chronology
        try:
            response_obj.additional_kwargs = getattr(response_obj, "additional_kwargs", {}) or {}
            response_obj.additional_kwargs["thinking"] = parsed_output.get("thinking", "")
        except Exception:
            pass
        return {
            # Append new AI message to the running conversation for this turn
            "messages": state["messages"] + [response_obj],
            "working_memory": state["working_memory"],
            # Keep last-pass thinking for compatibility (not relied on for trace)
            "thinking": parsed_output.get("thinking", "")
        }
        
    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Executes one or more tool calls in the last assistant message, sequentially."""
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []

        if not tool_calls:
            return {"working_memory": state.get("working_memory", ""), "messages": state["messages"]}

        # We'll mutate this as each tool applies
        working_mem = state.get("working_memory", "")
        out_msgs = list(state["messages"])

        for tool_call in tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            tool_args = dict(tool_call["args"] or {})
            tool_args["current_memory"] = working_mem  # inject the latest memory

            if tool.name == "overwrite_memory":
                try:
                    new_memory = tool.invoke(tool_args)
                except Exception as e:
                    tool_message = ToolMessage(
                        content=f"overwrite_memory failed: {e}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                    out_msgs.append(tool_message)
                    continue

                tool_message = ToolMessage(
                    content="overwrite_memory successfully applied.",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
                out_msgs.append(tool_message)
                working_mem = new_memory

            if tool.name == "delete_from_memory":
                tool_args["current_memory"] = working_mem

                try:
                    new_memory = tool.invoke(tool_args)
                except Exception as e:
                    tool_message = ToolMessage(
                        content=f"delete_from_memory failed: {e}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                    out_msgs.append(tool_message)
                    continue

                tool_message = ToolMessage(
                    content="delete_from_memory successfully applied.",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
                out_msgs.append(tool_message)
                working_mem = new_memory

            if tool.name == "append_in_memory":
                tool_args["current_memory"] = working_mem

                try:
                    new_memory = tool.invoke(tool_args)
                except Exception as e:
                    tool_message = ToolMessage(
                        content=f"append_in_memory failed: {e}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                    out_msgs.append(tool_message)
                    continue

                tool_message = ToolMessage(
                    content="append_in_memory successfully applied.",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
                out_msgs.append(tool_message)
                working_mem = new_memory

            if tool.name in ("patch_memory", "replace_in_memory"):
                tool_args["current_memory"] = working_mem

                try:
                    result = tool.invoke(tool_args)
                except Exception as e:
                    tool_message = ToolMessage(
                        content=f"{tool.name} failed: {e}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                    out_msgs.append(tool_message)
                    continue

                if isinstance(result, dict) and "new_memory" in result:
                    new_memory = result["new_memory"]
                elif isinstance(result, str):
                    new_memory = result
                else:
                    tool_message = ToolMessage(
                        content=f"{tool.name} returned unexpected result shape; no changes applied.",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                    out_msgs.append(tool_message)
                    continue

                tool_message = ToolMessage(
                    content=f"{tool.name} successfully applied.",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
                out_msgs.append(tool_message)
                working_mem = new_memory

            # Web search (pure)
            if tool.name == "web_search":
                result = tool.invoke(tool_args)
                tool_message = ToolMessage(
                    content=result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
                out_msgs.append(tool_message)
                working_mem = result

            # Code interpreter (updates sandbox_files)
            if tool.name == "code_interpreter":
                observation = tool.invoke(tool_args)
                content_str = format_e2b_output_to_str(observation)
                tool_message = ToolMessage(
                    content=content_str,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
                out_msgs.append(tool_message)
                working_mem = observation.get("files", [])

            # Fallback
            # result = tool.invoke(tool_args)
            # tool_message = ToolMessage(
            #     content=str(result),
            #     name=tool_call["name"],
            #     tool_call_id=tool_call["id"],
            # )
            # out_msgs.append(tool_message)
            # working_mem = result

        return {
            "messages": out_msgs,
            "working_memory": working_mem,
        }

    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        """
        Runs the ReAct loop for a single turn using a temporary thread for
        isolation, while persisting the working_memory in the 'main_thread'.
        """
        self.turn_counter += 1
        # Create a temporary, unique thread for this specific turn's execution
        turn_thread_config = {"configurable": {"thread_id": f"react_turn_{self.turn_counter}"}}
        main_thread_config = {"configurable": {"thread_id": "main_thread"}}
        
        # 1. Get the last known working memory (and sandbox files, if any) from the persistent main thread
        try:
            current_state = self.get_state()  # Gets state from "main_thread"
            current_working_memory = current_state.get("working_memory", INITIAL_WORKING_MEMORY)
            current_sandbox_files = current_state.get("sandbox_files", [])
        except Exception:
            current_working_memory = INITIAL_WORKING_MEMORY
            current_sandbox_files = []
        
        # 2. Invoke the graph in the temporary thread.
        #    This prevents state conflicts between complex ReAct loops.
        initial_state: Dict[str, Any] = {"messages": messages, "working_memory": current_working_memory}
        if self._has_code_tool:
            initial_state["sandbox_files"] = current_sandbox_files
        final_state = self.workflow.invoke(initial_state, config=turn_thread_config)
        
        # 3. Persist pruned conversation for next turn (exclude tool calls and tool messages from this turn)
        full_msgs: List[BaseMessage] = final_state["messages"]
        # Keep all prior messages that came in as input to this turn
        base_len = len(messages)
        prior_msgs = full_msgs[:base_len]
        turn_msgs = full_msgs[base_len:]

        # Keep only the final AI message (no tool_calls) from this turn
        pruned_turn_msgs: List[BaseMessage] = []
        for m in reversed(turn_msgs):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                pruned_turn_msgs = [m]
                break

        persisted_messages = prior_msgs + pruned_turn_msgs

        update_payload: Dict[str, Any] = {
            "working_memory": final_state.get("working_memory", INITIAL_WORKING_MEMORY),
            "messages": persisted_messages,
        }
        if self._has_code_tool:
            update_payload["sandbox_files"] = final_state.get("sandbox_files", [])
        self.workflow.update_state(main_thread_config, update_payload)
        
        # 4. Extract the final response for the user
        final_response = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_response = msg.content
                break
        
        # 5. Build a within-turn thinking/tool trace using only the new messages produced this turn
        delta_messages = turn_msgs
        trace_lines: List[str] = []
        for m in delta_messages:
            if isinstance(m, AIMessage):
                thought = getattr(m, "additional_kwargs", {}).get("thinking", "")
                if thought:
                    trace_lines.append("---Explicit Thought---")
                    trace_lines.append(thought)
                # Preserve raw content to keep pre-tool reasoning visible
                if m.content:
                    trace_lines.append(str(m.content))
                # Also show structure for debugging
                trace_lines.append(m.pretty_repr())
            elif isinstance(m, ToolMessage):
                trace_lines.append(m.pretty_repr())
        full_thinking_trace = "\n".join(trace_lines)
        
        return {"response": final_response, "thinking": full_thinking_trace}

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieves the latest state snapshot for the main thread and returns
        the actual state dictionary from its .values attribute.
        """
        snapshot = self.workflow.get_state({"configurable": {"thread_id": "main_thread"}})
        # The state dictionary is stored in the .values attribute of the snapshot
        if snapshot:
            return snapshot.values
        return {} # Return an empty dict if no state exists

    def get_private_state(self) -> str:
        state = self.get_state()
        memory = state.get('working_memory', '')
        return memory

    def reset(self) -> None:
        """Resets the agent by clearing the main thread's state."""
        empty_state: AgentState = AgentState(messages=[], working_memory=INITIAL_WORKING_MEMORY, thinking="")
        if self._has_code_tool:
            empty_state["sandbox_files"] = []
        self.workflow.update_state({"configurable": {"thread_id": "main_thread"}}, empty_state)
        print("ReActMemAgent state has been reset.")

    def close(self) -> None:
        """Optional resource cleanup; no-op by default when tools are injected externally."""
        return


# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":
    CONFIG_PATH = "config/config.yaml"
    print("Is file readable: ", os.access(CONFIG_PATH, os.R_OK))
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    try:
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_32b_openrouter") # qwen3_14b_vllm_hermes or gpt_oss_20b_openrouter
        print("✅ LLM Provider loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Provider: {e}")
        exit()

    agent = ReActMemAgent(llm_provider=main_llm, strategy="patch_and_replace")
    print("🤖 ReActMemAgent is ready. Type 'quit', 'exit', or 'q' to end.")

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
        print(agent.get_private_state())
        # Print thinking trace if available
        if "thinking" in output and output["thinking"]:
            print("\n---THINKING TRACE---")
            print(output["thinking"])

        print("\n---WORKING MEMORY---")
        print(agent.get_private_state())
        print("\n" + "="*50 + "\n")