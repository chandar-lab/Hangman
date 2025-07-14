# agent.py

import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# agent.py

import os
from typing import List, Dict, Any
from typing_extensions import TypedDict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph

# --- 1. Load Prompts ---
from prompts import DISTILLATION_SYSTEM_PROMPT, MAIN_SYSTEM_PROMPT

# --- 2. Setup and API Keys ---
# Make sure to set the ANTHROPIC_API_KEY environment variable if using Anthropic
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- 3. Model Configuration ---
# Choose your desired models here.
# 'model_provider' can be 'anthropic', or 'qwen'.
# 'model_name' is the specific model identifier.
MAIN_MODEL_CONFIG = {
    "model_provider": "qwen",  # 'anthropic' or 'qwen'
    "model_name": "Qwen/Qwen3-14B",
}

DISTILLATION_MODEL_CONFIG = {
    "model_provider": "qwen",  # 'anthropic' or 'qwen'
    "model_name": "Qwen/Qwen3-14B",
}


# --- 4. LLM Provider Class ---
# This class wraps different LLM backends (Anthropic API, local Transformers)
# and provides a unified interface for the agent.

from openai import OpenAI

class LLMProvider:
    """A wrapper for different language model providers."""

    def __init__(self, model_provider: str, model_name: str, **kwargs):
        self.model_provider = model_provider
        self.model_name = model_name
        self.kwargs = kwargs
        self.client = None # For API-based models
        self.model = None # For library-based models like Anthropic

        if self.model_provider == 'anthropic':
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is not set.")
            self.model = ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=self.model_name, **self.kwargs)

        elif self.model_provider == 'qwen':
            # Initialize the OpenAI client to point to the local vLLM server
            self.client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="not-needed"
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def invoke(self, messages: List[BaseMessage], thinking: bool = False) -> Dict[str, str]:
        """Invokes the model and returns the response and thinking trace."""
        if self.model_provider == 'anthropic':
            return self._invoke_anthropic(messages, thinking)
        elif self.model_provider == 'qwen':
            return self._invoke_qwen_api(messages, thinking)

    def _invoke_anthropic(self, messages: List[BaseMessage], thinking: bool) -> Dict[str, str]:
        """Handles invocation for Anthropic models."""
        self.model.thinking = thinking
        response = self.model.invoke(messages)
        thinking_trace, final_response = "", ""

        if thinking and isinstance(response.content, list) and len(response.content) > 1:
            thinking_block = next((b for b in response.content if b.get("type") == "thinking"), None)
            if thinking_block:
                thinking_trace = thinking_block.get("thinking", "")
            
            text_block = next((b for b in response.content if b.get("type") == "text"), None)
            if text_block:
                final_response = text_block.get("text", "")
        else:
            final_response = response.content if isinstance(response.content, str) else str(response.content)
            
        return {"response": final_response, "thinking": thinking_trace}

    def _invoke_qwen_api(self, messages: List[BaseMessage], thinking: bool = False) -> Dict[str, str]:
        """Handles invocation for Qwen models via a vLLM API endpoint."""
        # 1. Convert LangChain messages to the OpenAI dictionary format
        qwen_messages = [{"role": "user" if m.type == "human" else m.type, "content": m.content} for m in messages]
        
        # 2. Call the vLLM server
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=qwen_messages,
                max_tokens=4096,
                temperature=0.7,
            )
            full_response_text = completion.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while calling the vLLM server: {e}")
            return {"response": "Error: Could not connect to the local model server.", "thinking": ""}

        # 3. Manually parse the thinking block from the response string
        thinking_content = ""
        final_content = full_response_text

        think_start_tag = "<think>"
        think_end_tag = "</think>"
        start_index = full_response_text.find(think_start_tag)
        end_index = full_response_text.find(think_end_tag)

        # Check if a valid <think>...</think> block exists
        if start_index != -1 and end_index > start_index:
            thinking_content = full_response_text[start_index + len(think_start_tag):end_index].strip()
            final_content = full_response_text[end_index + len(think_end_tag):].strip()

        # 4. Return the parsed response
        # If thinking was not requested, we still strip the think block but return an empty thinking trace.
        if not thinking:
            return {"response": final_content, "thinking": ""}
        else:
            return {"response": final_content, "thinking": thinking_content}


# --- 5. Define Graph State ---
class AgentState(TypedDict):
    history: List[BaseMessage]
    working_memory: str
    response: str
    thinking: str
    diff: str

# --- 6. Define Graph Nodes ---

def generate_response(state: AgentState):
    """Generates a response to the user using the main configured model."""
    print("---NODE: GENERATING RESPONSE---")
    
    main_llm = LLMProvider(**MAIN_MODEL_CONFIG)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", MAIN_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
    ]).format(working_memory=state["working_memory"], history=state["history"])

    # Convert formatted prompt back to a list of messages if needed by provider
    # For now, we assume the provider's invoke method can handle the prompt string
    # or that we can reconstruct the message list easily. Let's pass the state history directly.
    
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
    """Parses the diff and applies the changes to the working memory."""
    print("---NODE: APPLYING DIFF---")
    
    diff = state.get("diff", "")
    if not diff:
        return {"working_memory": state["working_memory"]}

    updated_memory = state["working_memory"].splitlines()
    diff_lines = diff.strip().splitlines()

    for line in diff_lines:
        line = line.strip()
        if line.startswith("+ "):
            updated_memory.append(line[2:])
        elif line.startswith("~ "):
            # V1 implementation: Append the modification. A more advanced version
            # could implement a find-and-replace logic.
            updated_memory.append(line[2:])

    return {"working_memory": "\n".join(updated_memory)}


# --- 7. Build the Graph ---

workflow = StateGraph(AgentState)
workflow.add_node("generate_response", generate_response)
workflow.add_node("generate_diff", generate_diff)
workflow.add_node("apply_diff", apply_diff)

workflow.set_entry_point("generate_response")
workflow.add_edge("generate_response", "generate_diff")
workflow.add_edge("generate_diff", "apply_diff")
workflow.add_edge("apply_diff", END)

agent = workflow.compile()


# --- 8. Run the Agent ---
if __name__ == "__main__":
    print("Agent is ready. Type 'quit', 'exit', or 'q' to end the session.")
    print(f"Using Main Model: {MAIN_MODEL_CONFIG['model_provider']} ({MAIN_MODEL_CONFIG['model_name']})")
    print(f"Using Distillation Model: {DISTILLATION_MODEL_CONFIG['model_provider']} ({DISTILLATION_MODEL_CONFIG['model_name']})")
    
    initial_memory = ""
    config = {"recursion_limit": 5}
    
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
        
        result = agent.invoke(inputs, config=config)
        
        # Update state, but strip the system message from the history to avoid duplication
        state['working_memory'] = result['working_memory']
        # The history from the result already contains the new user message and AI response
        # so we just need to remove the prepended system message
        state['history'] = [msg for msg in result['history'] if msg.type not in ['system']]
        
        print("\n--------------------\n")
        print(f"AI: {result['response']}")
        print("\n---UPDATED WORKING MEMORY---")
        print(result["working_memory"])
        print("\n--------------------\n")