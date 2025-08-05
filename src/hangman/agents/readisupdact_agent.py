import os
import yaml
import json
from typing import List, Any, Dict

# --- Core LangChain/LangGraph Imports ---
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# --- Project-Specific Imports ---
from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
# Import prompts from the new central location for this agent
from hangman.prompts.readisupdact_agent import MAIN_SYSTEM_PROMPT, DISTILLATION_SYSTEM_PROMPT

# --- Agent State Definition ---

class AgentState(TypedDict):
    """
    Represents the complete state of the ReaDisUpdActAgent.
    """
    messages: List[BaseMessage]
    working_memory: str
    response: str
    thinking: str
    # A dictionary like {"deletions": [1], "insertions": ["new item"]}
    update_command: Dict[str, Any]

class ReaDisUpdActAgent(BaseAgent):
    """
    An agent that uses a "Reason-Distill-Update-Act" cycle. It distills
    the conversation into a JSON command to update its working memory.
    """
    def __init__(self, main_llm_provider: LLMProvider, distillation_llm_provider: LLMProvider):
        self.distillation_llm_provider = distillation_llm_provider
        super().__init__(llm_provider=main_llm_provider)
        self.turn_counter = 0
        self.reset()

    def _build_workflow(self) -> StateGraph:
        """Constructs the agent's LangGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("generate_update", self._generate_update)
        workflow.add_node("apply_update", self._apply_update)

        workflow.set_entry_point("generate_response")
        workflow.add_edge("generate_response", "generate_update")
        workflow.add_edge("generate_update", "apply_update")
        workflow.add_edge("apply_update", END)

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
        
        return {"thinking": result["thinking"], "response": result["response"]}

    def _generate_update(self, state: AgentState) -> dict:
        """Generates a JSON update command to modify the working memory."""
        print("---NODE: GENERATING UPDATE COMMAND---")
        messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
        
        prompt_str = DISTILLATION_SYSTEM_PROMPT.format(
            working_memory=state["working_memory"],
            messages=messages_str,
            thinking=state["thinking"],
            response=state["response"],
        )
        
        result = self.distillation_llm_provider.invoke([HumanMessage(content=prompt_str)])
        
        try:
            # The LLM is expected to return a JSON string. We parse it here.
            # Find the JSON block in case the LLM adds extra text
            json_str = result["response"].split('```json\n', 1)[-1].split('```')[0]
            update_command = json.loads(json_str)
            print(f"Parsed Update Command: {update_command}")
            return {"update_command": update_command}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing update command from LLM: {e}")
            return {"update_command": {"deletions": [], "insertions": []}} # Return empty command on error

    def _apply_update(self, state: AgentState) -> dict:
        """Parses the JSON command and applies it to the working memory."""
        print("---NODE: APPLYING UPDATE---")
        command = state.get("update_command", {})
        deletions = command.get("deletions", [])
        insertions = command.get("insertions", [])
        current_memory = state.get("working_memory", "")

        if not deletions and not insertions:
            return {} # No changes to apply

        new_memory_items = []
        if current_memory:
            existing_items = [line.split(". ", 1)[1] for line in current_memory.strip().split("\n") if ". " in line]
            for i, item in enumerate(existing_items):
                if (i + 1) not in deletions:
                    new_memory_items.append(item)
        
        new_memory_items.extend(insertions)
        
        new_memory = "\n".join(f"{i+1}. {item}" for i, item in enumerate(new_memory_items))
        print(f"\n--- WORKING MEMORY UPDATED ---")
        print(new_memory)
        return {"working_memory": new_memory}

    # --- Method Implementations from BaseAgent ---
    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        self.turn_counter += 1
        thread_config = {"configurable": {"thread_id": f"turn_{self.turn_counter}"}}
        main_thread_config = {"configurable": {"thread_id": "main_thread"}}

        try:
            current_state = self.get_state()
            current_working_memory = current_state.get("working_memory", "")
        except:
            current_working_memory = ""

        initial_state = {
            "messages": messages,
            "working_memory": current_working_memory,
            "response": "", "thinking": "", "update_command": {}
        }

        final_state = self.workflow.invoke(initial_state, config=thread_config)

        self.workflow.update_state(main_thread_config, {
            "messages": messages + [AIMessage(content=final_state["response"])],
            "working_memory": final_state["working_memory"]
        })
        return {"response": final_state["response"], "thinking": final_state["thinking"]}

    def get_state(self) -> AgentState:
        snapshot = self.workflow.get_state({"configurable": {"thread_id": "main_thread"}})
        return snapshot.values if snapshot else {}

    def get_private_state(self) -> str:
        state_values = self.get_state()
        memory = state_values.get('working_memory', 'N/A')
        thought = state_values.get('thinking', 'N/A')
        return f"---THINKING---\n{thought}\n\n---WORKING MEMORY---\n{memory}"

    def reset(self) -> None:
        thread_config = {"configurable": {"thread_id": "main_thread"}}
        empty_state = AgentState(messages=[], working_memory="", response="", thinking="", update_command={})
        self.workflow.update_state(thread_config, empty_state)
        print("Agent state has been reset.")


# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    try:
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local")
        distill_llm = load_llm_provider(CONFIG_PATH, provider_name="qwen3_14b_local")
        print("✅ LLM Providers loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Providers: {e}")
        exit()

    agent = ReaDisUpdActAgent(main_llm_provider=main_llm, distillation_llm_provider=distill_llm)
    print("🤖 ReaDisUpdActAgent Agent is ready. Type 'quit', 'exit', or 'q' to end.")

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
        print("\n---UPDATED WORKING MEMORY---")
        current_state = agent.get_state()
        print(current_state.get('working_memory', ''))
        print("\n" + "="*50 + "\n")