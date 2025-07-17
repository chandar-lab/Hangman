import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from diff_match_patch import diff_match_patch
from typing import List, Dict, Any
from typing_extensions import TypedDict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph

# --- LLM Provider Class ---
# This class wraps different LLM backends (Anthropic API, local Model)
# and provides a unified interface for the agent.

class LLMProvider:
    """A wrapper for different language model providers."""

    def __init__(self, model_provider: str, model_name: str, **kwargs):
        self.model_provider = model_provider
        self.model_name = model_name
        self.kwargs = kwargs
        self.client = None # For API-based models
        self.model = None # For library-based models like Anthropic

        if self.model_provider == 'anthropic':
            try:
                ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
            except:
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
            return self._invoke_qwen(messages, thinking)

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

    def _invoke_qwen(self, messages: List[BaseMessage], thinking: bool = False) -> Dict[str, str]:
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
