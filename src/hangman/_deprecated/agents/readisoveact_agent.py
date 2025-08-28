import os
import yaml
from typing import List, Any, Dict

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# --- Project-Specific Imports ---
# Import the abstract base class and state definitions
from hangman.agents.base_agent import BaseAgent, ModelOutput
# Import the unified LLM provider
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
# Import prompts from the new central location for this agent
from hangman.prompts.readisoveact_agent import MAIN_SYSTEM_PROMPT, DISTILLATION_SYSTEM_PROMPT

# --- Agent State Definition ---

class AgentState(TypedDict):
    """
    Represents the complete state of an agent at any point in time.
    This structure is used by the LangGraph state machine.
    """
    # The public conversation messages
    messages: List[BaseMessage]
    
    # The agent's private, internal knowledge and scratchpad
    working_memory: Dict[str, Any]
    
    # The public-facing response for the current turn
    response: str
    
    # The private thinking trace for the current turn
    thinking: str

class ReaDisOveActAgent(BaseAgent):
    """
    An agent that uses a "Reason-Distill-Overwrite-Act" cycle.
    It maintains a private working memory that is completely overwritten
    after each turn based on a distillation of the recent interaction.
    """
    def __init__(self, main_llm_provider: LLMProvider, distillation_llm_provider: LLMProvider):
        self.distillation_llm_provider = distillation_llm_provider
        super().__init__(llm_provider=main_llm_provider)
        self.turn_counter = 0
        self.reset()

    def _build_workflow(self) -> StateGraph:
        """Constructs the agent's LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Define the nodes for the new workflow
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("distill_and_overwrite_memory", self._distill_and_overwrite_memory)

        # Define the graph's structure
        workflow.set_entry_point("generate_response")
        workflow.add_edge("generate_response", "distill_and_overwrite_memory")
        workflow.add_edge("distill_and_overwrite_memory", END)

        return workflow.compile(checkpointer=MemorySaver())

    # --- Graph Nodes ---
    def _generate_response(self, state: AgentState) -> dict:
        """Generates a response using the main LLM provider."""
        print("---NODE: GENERATING RESPONSE---")
        prompt = ChatPromptTemplate.from_messages([
            ("system", MAIN_SYSTEM_PROMPT),
        ]).format(working_memory=state["working_memory"])
        
        messages = [SystemMessage(content=prompt)] + state["messages"]
        
        result = self.llm_provider.invoke(messages, thinking=True)
        
        return {
            "thinking": result["thinking"],
            "response": result["response"],
        }

    def _distill_and_overwrite_memory(self, state: AgentState) -> dict:
        """
        Generates a new working memory from scratch and overwrites the old one.
        """
        print("---NODE: DISTILLING & OVERWRITING MEMORY---")
        messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
        
        prompt_str = DISTILLATION_SYSTEM_PROMPT.format(
            working_memory=state["working_memory"],
            messages=messages_str,
            thinking=state["thinking"],
            response=state["response"],
        )
        
        # Use the distillation LLM to generate the *entire new* working memory
        result = self.distillation_llm_provider.invoke([HumanMessage(content=prompt_str)])
        
        new_memory = result["response"]
        print(f"\n--- WORKING MEMORY OVERWRITTEN ---")
        print(new_memory)
        
        # Return the new memory to directly overwrite the key in the state
        return {"working_memory": new_memory}

    # --- Method Implementations from BaseAgent ---
    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        self.turn_counter += 1
        thread_config = {"configurable": {"thread_id": f"turn_{self.turn_counter}"}}

        # Get current working memory from the main thread
        main_thread_config = {"configurable": {"thread_id": "main_thread"}}
        try:
            current_state = self.workflow.get_state(config=main_thread_config)
            current_working_memory = current_state.values.get("working_memory", "")
        except:
            current_working_memory = ""

        # Create initial state for the new thread, without the 'diff' key
        initial_state = {
            "messages": messages,
            "working_memory": current_working_memory,
            "response": "",
            "thinking": "",
        }

        # Invoke the graph with the initial state
        final_state = self.workflow.invoke(initial_state, config=thread_config)

        # Update the main thread with the results for persistence
        updated_messages = messages + [AIMessage(content=final_state["response"])]
        self.workflow.update_state(main_thread_config, {
            "messages": updated_messages,
            "working_memory": final_state["working_memory"]
        })

        return {"response": final_state["response"], "thinking": final_state["thinking"]}

    def get_state(self) -> AgentState:
        thread_config = {"configurable": {"thread_id": "main_thread"}}
        return self.workflow.get_state(config=thread_config).values
    
    def get_private_state(self) -> str:
        state_values = self.get_state()
        # Format the working memory and last thought into a loggable string
        memory = state_values.get('working_memory', 'N/A')
        # thought = state_values.get('thinking', 'N/A')
        return memory

    def reset(self) -> None:
        thread_config = {"configurable": {"thread_id": "main_thread"}}
        # Update empty state to remove the 'diff' key
        empty_state = AgentState(messages=[], working_memory="", response="", thinking="")
        self.workflow.update_state(thread_config, empty_state)
        print("Agent state has been reset.")


# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":
    # Load configuration from YAML file
    CONFIG_PATH = "config.yaml"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # --- Initialize Providers from Config ---
    try:
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local_vllm_native")
        distill_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local_vllm_native")
        print("✅ LLM Providers loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Providers: {e}")
        exit()

    # --- Initialize Agent ---
    agent = ReaDisOveActAgent(main_llm_provider=main_llm, distillation_llm_provider=distill_llm)
    print("🤖 ReaDisOveActAgent Agent is ready. Type 'quit', 'exit', or 'q' to end.")

    # --- Main Interaction Loop ---
    messages = []
    while True:
        user_input = input("User > ")
        if user_input.lower() in ["quit", 'exit', 'q']:
            print("Ending session.")
            break
        
        messages.append(HumanMessage(content=user_input))
        
        # Invoke the agent with the current messages
        output = agent.invoke(messages)
        
        # Update messages for the next turn
        messages.append(AIMessage(content=output["response"]))
        
        print("\n---ANSWER---")
        print(f"AI: {output['response']}")
        print("\n---UPDATED WORKING MEMORY---")
        current_state = agent.get_state()
        print(current_state['working_memory'])
        print("\n" + "="*50 + "\n")