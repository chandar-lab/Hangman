import os
import yaml
import json
from typing import List, Any, Dict, Sequence, Annotated

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# --- Project-Specific Imports ---
from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
# Assume prompts for this agent are in a dedicated file
from hangman.prompts.react_agent import MAIN_SYSTEM_PROMPT

# --- Tool Definition ---

@tool
def update_memory(deletions: List[int], insertions: List[str], current_memory: str) -> str:
    """
    Updates the working memory by deleting specified line numbers and appending new items.
    
    Args:
        deletions: A list of line numbers to delete from the memory. Numbers are 1-based.
        insertions: A list of new strings to add to the end of the memory.
        current_memory: The current working memory string.
    
    Returns:
        The newly formatted working memory string.
    """
    print(f"---TOOL: update_memory---")
    print(f"Deletions: {deletions}, Insertions: {insertions}")

    # Start with an empty list for the new memory
    new_memory_items = []
    
    # 1. Parse the current memory and keep only the items NOT marked for deletion
    if current_memory:
        existing_items = [line.split(". ", 1)[1] for line in current_memory.strip().split("\n") if ". " in line]
        for i, item in enumerate(existing_items):
            # Keep the item if its 1-based index is NOT in the deletions list
            if (i + 1) not in deletions:
                new_memory_items.append(item)

    # 2. Perform insertions by extending the list of kept items
    if insertions:
        new_memory_items.extend(insertions)
    
    # 3. Re-format the final list into a numbered list string
    new_memory = "\n".join(f"{i+1}. {item}" for i, item in enumerate(new_memory_items))
    print(f"New Memory:\n{new_memory}")
    return new_memory


# --- Agent State and Class Definition ---

class AgentState(TypedDict):
    """The state for the ReActAgent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thinking: str
    working_memory: str

class ReActAgent(BaseAgent):
    """
    An agent that uses the ReAct (Reason+Act) paradigm.
    Its only tool is the ability to edit its own working memory.
    """
    def __init__(self, main_llm_provider: LLMProvider):
        # This agent only needs one LLM.
        self.tools = [update_memory]
        self.tools_by_name = {t.name: t for t in self.tools}
        
        # Bind the tools to the LLM
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

    def _should_continue(self, state: AgentState):
        """Determines whether to continue with a tool call or end."""
        if not state["messages"][-1].tool_calls:
            return "end"
        return "continue"

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """Invokes the LLM with the current state to decide on an action or response."""
        print("---NODE: AGENT---")
        # Format the system prompt with the current working memory
        system_prompt = SystemMessage(content=MAIN_SYSTEM_PROMPT.format(working_memory=state["working_memory"]))
        messages = [system_prompt] + state["messages"]

        response_obj = self.model.invoke(messages)
        parsed_output = self.llm_provider.parse_response(response_obj.content or "")
        response_obj.content = parsed_output["response"]
        return {
            "messages": [response_obj],
            "working_memory": state["working_memory"],
            "thinking": parsed_output["thinking"]
        }
        
    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Executes tools and updates state."""
        tool_call = state["messages"][-1].tool_calls[0]
        tool = self.tools_by_name[tool_call["name"]]
        
        # The `update_memory` tool needs the current memory to modify it
        tool_args = tool_call["args"]
        tool_args["current_memory"] = state["working_memory"]

        new_memory = tool.invoke(tool_args)
        
        tool_message = ToolMessage(
            content=f"Successfully updated working memory.",
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )
        # Update both working memory and add the tool result message
        return {"working_memory": new_memory, "messages": [tool_message]}

    # --- Method Implementations from BaseAgent ---

    # def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
    #     """Runs the ReAct loop for a single turn."""
    #     self.turn_counter += 1
    #     main_thread_config = {"configurable": {"thread_id": "main_thread"}}
        
    #     # Get current working memory from the main persistent thread
    #     try:
    #         current_state = self.workflow.get_state(config=main_thread_config)
    #         current_working_memory = current_state.get("working_memory", "")
    #         print(f"--- Current Working Memory ---\n{current_working_memory}\n")
    #         # The ReAct loop needs the full message history
    #         current_messages = current_state.get("messages", [])
    #     except:
    #         current_working_memory = ""
    #         current_messages = []
        
    #     # The input for this turn is the accumulated history plus the new message
    #     input_messages = current_messages + messages

    #     # Invoke the graph. It will loop between agent and tools until it finishes.
    #     final_state = self.workflow.invoke(
    #         {"messages": input_messages, "working_memory": current_working_memory}, 
    #         config=main_thread_config
    #     )
        
    #     # Persist the final state back to the main thread
    #     self.workflow.update_state(main_thread_config, final_state)
        
    #     # Extract the final response and a trace of the tool calls for logging
    #     final_response = final_state["messages"][-1].content
    #     tool_trace = [msg.pretty_repr() for msg in final_state["messages"] if isinstance(msg, (AIMessage, ToolMessage))]
        
    #     return {"response": final_response}#, "thinking": "\n".join(tool_trace)}
    
    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        """
        Runs the ReAct loop for a single turn using a temporary thread for
        isolation, while persisting the working_memory in the 'main_thread'.
        """
        self.turn_counter += 1
        # Create a temporary, unique thread for this specific turn's execution
        turn_thread_config = {"configurable": {"thread_id": f"react_turn_{self.turn_counter}"}}
        main_thread_config = {"configurable": {"thread_id": "main_thread"}}
        
        # 1. Get the last known working memory from the persistent main thread
        try:
            current_state = self.get_state() # Gets state from "main_thread"
            current_working_memory = current_state.get("working_memory", "")
        except Exception:
            current_working_memory = ""
        
        # 2. Invoke the graph in the temporary thread.
        #    This prevents state conflicts between complex ReAct loops.
        final_state = self.workflow.invoke(
            {"messages": messages, "working_memory": current_working_memory}, 
            config=turn_thread_config
        )
        
        # 3. Explicitly save the updated working memory back to the main thread for the next turn.
        self.workflow.update_state(
            main_thread_config,
            {"working_memory": final_state["working_memory"], "messages": final_state["messages"]}
        )
        
        # 4. Extract the final response for the user
        final_response = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_response = msg.content
                break
        
        # 5. Create a comprehensive thinking trace
        explicit_thinking = final_state.get("thinking", "")
        tool_trace = [msg.pretty_repr() for msg in final_state["messages"] if isinstance(msg, (AIMessage, ToolMessage))]
        full_thinking_trace = f"---Explicit Thought---\n{explicit_thinking}\n\n---Tool Trace---\n" + "\n".join(tool_trace)
        
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
        memory = state.get('working_memory', 'N/A')
        return f"---WORKING MEMORY---\n{memory}"

    def reset(self) -> None:
        """Resets the agent by clearing the main thread's state."""
        empty_state = AgentState(messages=[], working_memory="")
        self.workflow.update_state({"configurable": {"thread_id": "main_thread"}}, empty_state)
        print("ReactAgent state has been reset.")


# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    try:
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local")
        print("✅ LLM Provider loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Provider: {e}")
        exit()

    agent = ReActAgent(main_llm_provider=main_llm)
    print("🤖 ReActAgent is ready. Type 'quit', 'exit', or 'q' to end.")

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
        print("\n" + "="*50 + "\n")