import os
import yaml
from typing import List, Any, Dict

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# --- Project-Specific Imports ---
from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.prompts.public_cot_agent import MAIN_SYSTEM_PROMPT

# --- Agent State and Class Definition ---

class AgentState(TypedDict):
    messages: List[BaseMessage]
    thinking: str

class PublicCoTAgent(BaseAgent):
    """
    A stateless agent that appends its chain-of-thought (CoT) reasoning to its public answer.
    The full reasoning is part of the public dialogue.
    """
    def __init__(self, main_llm_provider: LLMProvider):
        self.model = main_llm_provider.client
        super().__init__(llm_provider=main_llm_provider)
        self.reset()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._call_model)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)
        return workflow.compile()

    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        system_prompt = SystemMessage(content=MAIN_SYSTEM_PROMPT)
        messages = [system_prompt] + state["messages"]
        response_obj = self.model.invoke(messages)
        parsed_output = self.llm_provider.parse_response(response_obj.content or "")
        # Append thinking to the response, both public
        public_response = parsed_output["response"]
        thinking = parsed_output.get("thinking", "")
        if thinking:
            public_response = f"{public_response}\n\n---Chain-of-Thought---\n{thinking}"
        response_obj.content = public_response
        return {
            "messages": [response_obj],
            "thinking": thinking
        }

    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        final_state = self.workflow.invoke({"messages": messages, "thinking": ""})
        final_response = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage):
                final_response = msg.content
                break
        thinking = final_state.get("thinking", "")
        return {"response": final_response, "thinking": thinking}

    def get_state(self) -> Dict[str, Any]:
        # Stateless: always returns empty dict
        return {}

    def get_private_state(self) -> str:
        return ""

    def reset(self) -> None:
        # No persistent state to reset
        pass

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
    agent = PublicCoTAgent(main_llm_provider=main_llm)
    print("🤖 PublicCoTAgent is ready. Type 'quit', 'exit', or 'q' to end.")
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
        print("\n" + "="*50 + "\n")
