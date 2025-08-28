import os
import yaml
from typing import List, Any, Dict
from diff_match_patch import diff_match_patch

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
# Import prompts from a central location
from hangman.prompts.readispatact_agent import MAIN_SYSTEM_PROMPT, DISTILLATION_SYSTEM_PROMPT

# --- Agent State Definition ---

class AgentState(TypedDict):
    """
    Represents the complete state of an agent at any point in time.
    This structure is used by the LangGraph state machine.
    """
    # The public conversation messages
    messages: List[BaseMessage]
    
    # The agent's private, internal knowledge and scratchpad
    working_memory: str
    
    # The public-facing response for the current turn
    response: str
    
    # The private thinking trace for the current turn
    thinking: str

    # The diff of changes made to the working memory
    diff: str

class ReaDisPatActAgent(BaseAgent):
    """
    An agent that uses a "think-distill-patch" cycle to maintain a
    private working memory separate from its conversational messages.
    """
    def __init__(self, main_llm_provider: LLMProvider, distillation_llm_provider: LLMProvider):
        self.distillation_llm_provider = distillation_llm_provider
        # The parent class __init__ will call _build_workflow and assign it to self.workflow
        super().__init__(llm_provider=main_llm_provider)
        # The self.workflow object now manages all state. No separate tracker is needed.
        self.turn_counter = 0
        self.reset()

    def _build_workflow(self) -> StateGraph:
        """Constructs the agent's LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Nodes are now methods of the class, giving them access to self.
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("generate_diff", self._generate_diff)
        workflow.add_node("apply_diff", self._apply_diff)

        workflow.set_entry_point("generate_response")
        workflow.add_edge("generate_response", "generate_diff")
        workflow.add_edge("generate_diff", "apply_diff")
        workflow.add_edge("apply_diff", END)

        return workflow.compile(checkpointer=MemorySaver())

    # --- Graph Nodes ---
    def _generate_response(self, state: AgentState) -> dict:
        """Generates a response using the main LLM provider."""
        print("---NODE: GENERATING RESPONSE---")
        # Format the prompt with the current working memory
        prompt = ChatPromptTemplate.from_messages([
            ("system", MAIN_SYSTEM_PROMPT),
        ]).format(working_memory=state["working_memory"])
        
        messages = [SystemMessage(content=prompt)] + state["messages"]
        
        # Use the main LLM provider passed during initialization
        result = self.llm_provider.invoke(messages, thinking=True)
        
        return {
            "thinking": result["thinking"],
            "response": result["response"],
        }

    def _generate_diff(self, state: AgentState) -> dict:
        """Generates a diff to update the working memory."""
        print("---NODE: GENERATING DIFF---")
        messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
        
        prompt_str = DISTILLATION_SYSTEM_PROMPT.format(
            working_memory=state["working_memory"],
            messages=messages_str,
            thinking=state["thinking"],
            response=state["response"],
        )
        
        # Use the separate distillation LLM provider
        result = self.distillation_llm_provider.invoke([HumanMessage(content=prompt_str)])
        return {"diff": result["response"]}

    def _apply_diff(self, state: AgentState) -> dict:
        """Parses and applies a text patch to the working memory."""
        print("---NODE: APPLYING DIFF---")
        diff_text = state.get("diff", "").replace('```', "").strip()

        if not diff_text:
            return {}

        dmp = diff_match_patch()
        try:
            patches = dmp.patch_fromText(diff_text)
            new_memory, _ = dmp.patch_apply(patches, state["working_memory"])
            # print(f"\n--- WORKING MEMORY UPDATED ---")
            # print(new_memory)
            return {"working_memory": new_memory}
        except Exception as e:
            print(f"Error applying patch: {e}")
            return {}

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

        # Create initial state for the new thread
        initial_state = {
            "messages": messages,
            "working_memory": current_working_memory,
            "response": "",
            "thinking": "",
            "diff": ""
        }

        # Invoke the graph with the initial state - this will execute the full workflow
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
        # To reset, we update the state with an empty AgentState for that thread
        empty_state = AgentState(messages=[], working_memory="", response="", thinking="", diff="")
        self.workflow.update_state(thread_config, empty_state)
        print("Agent state has been reset.")


# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":
    # Load configuration from YAML file
    CONFIG_PATH = "config.yaml"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # --- Initialize Providers from Config ---
    # This assumes you have providers named 'qwen_local' and 'kimi_k2_openrouter' in your config
    # You can change these names to match your config file.
    try:
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local_vllm_native")
        distill_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local_vllm_native")
        print("✅ LLM Providers loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Providers: {e}")
        exit()

    # --- Initialize Agent ---
    agent = ReaDisPatActAgent(main_llm_provider=main_llm, distillation_llm_provider=distill_llm)
    print("🤖 ReaDisPatActAgent Agent is ready. Type 'quit', 'exit', or 'q' to end.")

    # --- Main Interaction Loop ---
    messages = []
    while True:
        user_input = input("User > ")
        if user_input.lower() in ["quit", "exit", "q"]:
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