import os
import yaml
from typing import List

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# --- Project-Specific Imports ---
from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.prompts.private_cot_agent import MAIN_SYSTEM_PROMPT


class AgentState(TypedDict):
    """
    Represents the state for the PrivateCoTAgent.
    """
    # The public conversation messages
    messages: List[BaseMessage]

    # A string containing all past reasoning traces, separated by a delimiter.
    working_memory: str

    # The public-facing response for the current turn
    response: str

    # The private thinking trace for the current turn
    thinking: str


# A unique delimiter to separate thoughts in the working memory string
THOUGHT_DELIMITER = "\n<END_OF_THOUGHT>\n"


class PrivateCoTAgent(BaseAgent):
    """
    An agent that uses a "Private Chain-of-Thought" cycle.
    It generates a reasoning trace ("thinking"), acts on it, and then
    appends that trace to its working memory for future turns.
    """

    def __init__(self, main_llm_provider: LLMProvider):
        # This agent only needs one LLM provider.
        super().__init__(llm_provider=main_llm_provider)
        self.turn_counter = 0
        self.reset()

    def _build_workflow(self) -> StateGraph:
        """Constructs the agent's LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # The workflow is a simple 2-step process
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("keep_reasoning_trace", self._keep_reasoning_trace)

        workflow.set_entry_point("generate_response")
        workflow.add_edge("generate_response", "keep_reasoning_trace")
        workflow.add_edge("keep_reasoning_trace", END)

        return workflow.compile(checkpointer=MemorySaver())

    # --- Graph Nodes ---
    def _generate_response(self, state: AgentState) -> dict:
        """
        Generates a response by first interleaving past thoughts into the
        message history to provide full context to the LLM.
        """
        print("---NODE: GENERATING RESPONSE---")

        # Reconstruct the prompt with interleaved reasoning
        past_thoughts = state["working_memory"].split(THOUGHT_DELIMITER)
        interleaved_messages: List[BaseMessage] = []
        thought_idx = 0

        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                interleaved_messages.append(msg)
            elif isinstance(msg, AIMessage):
                # Before each AI message, insert its corresponding past thought
                if thought_idx < len(past_thoughts) and past_thoughts[thought_idx]:
                    thought_content = f"<thinking>{past_thoughts[thought_idx]}</thinking>"
                    # Represent the agent's past turn as a single AIMessage
                    # containing both its thought and its final answer.
                    interleaved_content = f"{thought_content}\n{msg.content}"
                    interleaved_messages.append(AIMessage(content=interleaved_content))
                else:
                    # If there's no corresponding thought, just add the message
                    interleaved_messages.append(msg)
                thought_idx += 1

        messages = [SystemMessage(content=MAIN_SYSTEM_PROMPT)] + interleaved_messages

        # Use the main LLM provider to generate a new thought and response
        result = self.llm_provider.invoke(messages, thinking=True)

        return {
            "thinking": result["thinking"],
            "response": result["response"],
        }

    def _keep_reasoning_trace(self, state: AgentState) -> dict:
        """Appends the most recent thought to the working memory string."""
        print("---NODE: KEEPING REASONING TRACE---")

        new_thought = state.get("thinking", "").strip()
        if not new_thought:
            return {}

        # Append the new thought using the delimiter
        if state["working_memory"]:
            new_working_memory = state["working_memory"] + THOUGHT_DELIMITER + new_thought
        else:
            new_working_memory = new_thought

        print(f"\n--- WORKING MEMORY APPENDED ---")
        return {"working_memory": new_working_memory}

    # --- Method Implementations from BaseAgent ---
    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        self.turn_counter += 1
        thread_config = {"configurable": {"thread_id": f"turn_{self.turn_counter}"}}

        # Get current working memory from the main thread
        main_thread_config = {"configurable": {"thread_id": "main_thread"}}
        try:
            current_state = self.workflow.get_state(config=main_thread_config)
            current_working_memory = current_state.values.get("working_memory", "")
        except Exception:
            current_working_memory = ""

        # Create initial state for the new thread
        initial_state = {
            "messages": messages,
            "working_memory": current_working_memory,
            "response": "",
            "thinking": "",
        }

        # Invoke the graph with the initial state
        final_state = self.workflow.invoke(initial_state, config=thread_config)

        # Update the main thread with the results for persistence
        self.workflow.update_state(
            main_thread_config,
            {
                "messages": messages + [AIMessage(content=final_state["response"])],
                "working_memory": final_state["working_memory"],
            },
        )

        return {"response": final_state["response"], "thinking": final_state["thinking"]}

    def get_state(self) -> AgentState:
        thread_config = {"configurable": {"thread_id": "main_thread"}}
        return self.workflow.get_state(config=thread_config).values

    def get_private_state(self) -> str:
        state_values = self.get_state()
        memory = state_values.get("working_memory", "")
        # thought = state_values.get('thinking', 'N/A') ## TODO: This does not work
        return memory

    def reset(self) -> None:
        thread_config = {"configurable": {"thread_id": "main_thread"}}
        empty_state = AgentState(messages=[], working_memory="", response="", thinking="")
        self.workflow.update_state(thread_config, empty_state)
        print("Agent state has been reset.")


# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    try:
        # This agent only needs one LLM.
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local_vllm_native")
        print("✅ LLM Provider loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Provider: {e}")
        raise SystemExit(1)

    agent = PrivateCoTAgent(main_llm_provider=main_llm)
    print("🤖 PrivateCoTAgent is ready. Type 'quit', 'exit', or 'q' to end.")

    messages: List[BaseMessage] = []
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
        print("\n---UPDATED WORKING MEMORY---")
        current_state = agent.get_state()
        print(current_state["working_memory"])
        print("\n" + "=" * 50 + "\n")
