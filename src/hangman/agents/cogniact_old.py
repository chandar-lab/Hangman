# agent.py

import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from diff_match_patch import diff_match_patch
from typing import List, Dict, Any
from typing_extensions import TypedDict

from hangman.providers.llmprovider import LLMProvider

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

# --- Load Prompts and Define Model Configs ---
from hangman.prompts import DISTILLATION_SYSTEM_PROMPT,  MAIN_SYSTEM_PROMPT

MAIN_MODEL_CONFIG = {
    "model_provider": "qwen",  # 'anthropic' or 'qwen'
    "model_name": "Qwen/Qwen3-14B",
}

DISTILLATION_MODEL_CONFIG = {
    "model_provider": "qwen",  # 'anthropic' or 'qwen'
    "model_name": "Qwen/Qwen3-14B",
}

# --- Define Graph State ---
class AgentState(TypedDict):
    history: List[BaseMessage]
    working_memory: str
    response: str
    thinking: str
    diff: str

# --- Define Graph Nodes ---

def generate_response(state: AgentState):
    """Generates a response to the user using the main configured model."""
    print("---NODE: GENERATING RESPONSE---")
    
    main_llm = LLMProvider(**MAIN_MODEL_CONFIG)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", MAIN_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
    ]).format(working_memory=state["working_memory"], history=state["history"])

    result = main_llm.invoke(state["history"], thinking=True)
    
    ai_message = AIMessage(content=result["response"])
    update = {
        "history": state["history"] + [ai_message],
        "thinking": result["thinking"],
        "response": result["response"],
    }
    return update

def generate_diff(state: AgentState):
    """Generates a diff to update the working memory using the distillation model."""
    print("---NODE: GENERATING DIFF---")
    
    distillation_llm = LLMProvider(**DISTILLATION_MODEL_CONFIG, temperature=0.0)
    
    # Format the history for the prompt
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state["history"]])
    
    prompt_str = DISTILLATION_SYSTEM_PROMPT.format(
        working_memory=state["working_memory"],
        history=history_str,
        thinking=state["thinking"],
        response=state["response"],
    )
    
    # The distillation model doesn't need a complex message history, just the formatted prompt.
    result = distillation_llm.invoke([HumanMessage(content=prompt_str)], thinking=True)
    update = {"diff": result["response"]}
    return update

def apply_diff(state: AgentState):
    """Parses and applies a text patch to the working memory."""
    print("---NODE: APPLYING DIFF---")

    diff_text = state.get("diff", "")
    print(f"\nDiff Text:\n----\n{diff_text}\n----\n")
    diff_text = diff_text.replace('```', "").strip()
    working_memory = state["working_memory"]

    if not diff_text:
        return {"working_memory": working_memory}

    dmp = diff_match_patch()

    try:
        # The library expects to parse the patch from its text format
        patches = dmp.patch_fromText(diff_text)
        # Apply the patch to the current working memory
        new_memory, results = dmp.patch_apply(patches, working_memory)

        return {"working_memory": new_memory}

    except Exception as e:
        print(f"Error applying patch: {e}")
        return {"working_memory": working_memory}


# --- 7. Build the Graph ---

workflow = StateGraph(AgentState)
workflow.add_node("generate_response", generate_response)
workflow.add_node("generate_diff", generate_diff)
workflow.add_node("apply_diff", apply_diff)

workflow.set_entry_point("generate_response")
workflow.add_edge("generate_response", "generate_diff")
workflow.add_edge("generate_diff", "apply_diff")
workflow.add_edge("apply_diff", END)

cogniact = workflow.compile()


# --- 8. Run the Agent ---
if __name__ == "__main__":
    print("Agent is ready. Type 'quit', 'exit', or 'q' to end the session.")
    print(f"Using Main Model: {MAIN_MODEL_CONFIG['model_provider']} ({MAIN_MODEL_CONFIG['model_name']})")
    print(f"Using Distillation Model: {DISTILLATION_MODEL_CONFIG['model_provider']} ({DISTILLATION_MODEL_CONFIG['model_name']})")
    
    initial_memory = ""
    
    state = AgentState(
        history=[],
        working_memory=initial_memory,
        response="",
        thinking="",
        diff="",
    )

    while True:
        user_input = input("User > ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Ending session.")
            break
        
        # Prepare the input for the agent
        inputs = {
            # The history now includes the system prompt and the conversation
            "history": [SystemMessage(content=MAIN_SYSTEM_PROMPT.format(working_memory=state["working_memory"]))] + state["history"] + [HumanMessage(content=user_input)],
            "working_memory": state["working_memory"],
        }
        
        result = cogniact.invoke(inputs)
        
        # Update state, but strip the system message from the history to avoid duplication
        state['working_memory'] = result['working_memory']
        # The history from the result already contains the new user message and AI response
        # so we just need to remove the prepended system message
        state['history'] = [msg for msg in result['history'] if msg.type not in ['system']]
        
        print("\n---UPDATED WORKING MEMORY---")
        print(result["working_memory"])
        print("\n---ANSWER---\n")
        print(f"AI: {result['response']}")
        print("\n--------------------\n")