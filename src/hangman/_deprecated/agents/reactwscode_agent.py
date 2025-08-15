import os
import yaml
from typing import List, Any, Dict, Sequence, Annotated

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# --- Project-Specific Imports ---
from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
# --- Import ALL tools and the new prompt ---
from hangman.tools import (
    update_memory, 
    get_search_tool, 
    E2BCodeInterpreterTool, 
    format_e2b_output_to_str
)
from hangman.prompts.reactwscode_agent import MAIN_SYSTEM_PROMPT

# --- Agent State and Class Definition ---

class AgentState(TypedDict):
    """The state for the ReActWSCodeAgent."""
    messages: List[BaseMessage]
    thinking: str
    working_memory: str
    sandbox_files: List[str]  

class ReActWSCodeAgent(BaseAgent):
    """
    An agent that uses the ReAct paradigm with three tools:
    1. update_memory: to manage an internal scratchpad.
    2. web_search: to find up-to-date information online.
    3. code_interpreter: to execute Python code in a secure sandbox.
    """
    def __init__(self, main_llm_provider: LLMProvider, search_provider: str = "duckduckgo"):
        # 1. Instantiate all tools
        web_search_tool = get_search_tool(provider=search_provider)
        self.interpreter = E2BCodeInterpreterTool()
        code_interpreter_tool = self.interpreter.as_langchain_tool() 

        # 2. Assemble the list of available tools
        self.tools = [update_memory, web_search_tool, code_interpreter_tool]
        self.tools_by_name = {t.name: t for t in self.tools}
        
        # 3. Bind the tools to the LLM
        self.model = main_llm_provider.client.bind_tools(self.tools)

        super().__init__(llm_provider=main_llm_provider)
        self.turn_counter = 0 # Used for thread management
        self.reset()

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

    def _should_continue(self, state: AgentState) -> str:
        """Determines whether to continue with a tool call or end."""
        if not state["messages"][-1].tool_calls:
            return "end"
        return "continue"

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """Invokes the LLM with the current state to decide on an action or response."""
        print("---NODE: AGENT---")
        # Format the system prompt with the current working memory, using the new prompt
        system_prompt = SystemMessage(content=MAIN_SYSTEM_PROMPT.format(
            working_memory=state["working_memory"],
            sandbox_files=", ".join(state.get("sandbox_files", [])) if state.get("sandbox_files", []) else "None"
        ))
        messages = [system_prompt] + state["messages"]

        response_obj = self.model.invoke(messages)
        # We assume the LLM provider can parse out thinking/response content if structured
        parsed_output = self.llm_provider.parse_response(response_obj.content or "")
        response_obj.content = parsed_output["response"]
        
        return {
            "messages": [response_obj],
            "thinking": parsed_output["thinking"]
            # working_memory is not modified in this node
        }
        
    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the called tool and returns the result. This node is now generic
        and handles each tool's specific side effects.
        """
        tool_call = state["messages"][-1].tool_calls[0]
        tool = self.tools_by_name[tool_call["name"]]
        tool_args = tool_call["args"]

        print(f"---NODE: TOOL ({tool.name})---")

        if tool.name == "update_memory":
            # This tool has a side effect: it modifies working_memory.
            # We must inject the current memory state into its arguments.
            tool_args["current_memory"] = state["working_memory"]
            result = tool.invoke(tool_args)
            
            tool_message = ToolMessage(
                content=f"Successfully updated working memory.", # A confirmation message
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            # Return the updated memory AND the tool message
            return {"working_memory": result, "messages": [tool_message]}

        elif tool.name == "web_search":
            # This tool is pure; it takes a query and returns a result.
            # It does not modify the working_memory directly.
            result = tool.invoke(tool_args)
            
            tool_message = ToolMessage(
                content=result, # The content is the actual search result
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            # Only return the tool message. The agent must use update_memory
            # in a subsequent step if it wants to save the search results.
            return {"messages": [tool_message]}
        
        elif tool.name == "code_interpreter":
            # The tool's func returns a dictionary
            observation_dict = tool.invoke(tool_args)
            
            # Format the dict into a string for the LLM
            content_str = format_e2b_output_to_str(observation_dict)
            
            tool_message = ToolMessage(
                content=content_str,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            # Return the message AND the updated list of files for the state
            return {
                "messages": [tool_message], 
                "sandbox_files": observation_dict.get("files", [])
            }
        
        else:
            # Fallback for any other tools
            result = tool.invoke(tool_args)
            tool_message = ToolMessage(
                content=str(result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            return {"messages": [tool_message]}


    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        """
        Runs the ReAct loop for a single turn using a temporary thread for
        isolation, while persisting the working_memory in the 'main_thread'.
        """
        self.turn_counter += 1
        turn_thread_config = {"configurable": {"thread_id": f"react_web_search_turn_{self.turn_counter}"}}
        main_thread_config = {"configurable": {"thread_id": "main_thread"}}
        
        try:
            current_state = self.get_state()
            current_working_memory = current_state.get("working_memory", "")
            current_sandbox_files = current_state.get("sandbox_files", [])
        except Exception:
            current_working_memory = ""
            current_sandbox_files = []
        
        final_state = self.workflow.invoke(
            {
                "messages": messages, 
                "working_memory": current_working_memory, 
                "sandbox_files": current_sandbox_files
            },
            config=turn_thread_config
        )
        
        self.workflow.update_state(
            main_thread_config,
            {
                "working_memory": final_state["working_memory"], 
                "messages": final_state["messages"], 
                "sandbox_files": final_state["sandbox_files"],
            }
            
        )
        
        final_response = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_response = msg.content
                break
        
        explicit_thinking = final_state.get("thinking", "")
        tool_trace = [msg.pretty_repr() for msg in final_state["messages"] if isinstance(msg, (AIMessage, ToolMessage))]
        full_thinking_trace = f"---Explicit Thought---\n{explicit_thinking}\n\n---Tool Trace---\n" + "\n".join(tool_trace)
        
        return {"response": final_response, "thinking": full_thinking_trace}

    def get_state(self) -> Dict[str, Any]:
        """Retrieves the latest state snapshot from the main thread."""
        snapshot = self.workflow.get_state({"configurable": {"thread_id": "main_thread"}})
        return snapshot.values if snapshot else {}

    def get_private_state(self) -> str:
        state = self.get_state()
        memory = state.get('working_memory', 'N/A')
        return memory

    def reset(self) -> None:
        """Resets the agent by clearing the main thread's state."""
        empty_state = AgentState(messages=[], working_memory="", thinking="", sandbox_files=[]) # <-- ADD THIS
        self.workflow.update_state({"configurable": {"thread_id": "main_thread"}}, empty_state)
        print("ReActWSCodeAgent state has been reset.")
    
    def close(self):
        """Safely closes the code interpreter session by calling the method on the interpreter tool."""
        print("Closing agent's resources...")
        self.interpreter.close()


# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    try:
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="kimi_k2_openrouter")
        print("✅ LLM Provider loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Provider: {e}")
        exit()

    # Instantiate the new, fully-equipped agent
    agent = ReActWSCodeAgent(main_llm_provider=main_llm, search_provider="duckduckgo")
    print("🤖 ReActWSCodeAgent is ready. Type 'quit', 'exit', or 'q' to end.")
    print("Try asking it to plot something, e.g., 'Plot the sine function from -pi to pi'")

    try:
        messages = []
        while True:
            user_input = input("User > ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            
            messages.append(HumanMessage(content=user_input))
            
            output = agent.invoke(messages)
            
            messages.append(AIMessage(content=output["response"]))
            
            print("\n---ANSWER---")
            print(f"AI: {output['response']}")
            print(agent.get_private_state())
            if "thinking" in output and output["thinking"]:
                print("\n---THINKING TRACE---")
                print(output["thinking"])
            print("\n" + "="*50 + "\n")
            
    finally:
        # CRITICAL: This ensures the E2B sandbox is always closed, even if an error occurs.
        print("Cleaning up agent resources...")
        agent.close()
        print("Session ended.")