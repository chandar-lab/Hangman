import os
import yaml
import logging
from typing import List, Any, Literal, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Type Definitions ---
class ModelOutput(TypedDict):
    response: str
    thinking: str

class APIConfig(TypedDict):
    base_url: str
    api_key_env: str

class GenerationConfig(TypedDict):
    temperature: float
    max_tokens: int

class ProviderConfig(TypedDict):
    name: str
    provider_type: Literal["openai"]
    model_name: str
    # NEW: This key determines how to parse the model's output.
    parsing_format: Optional[Literal["think_tags", "direct_response"]]
    api_config: APIConfig
    generation_config: GenerationConfig

# --- LLM Provider ---
class LLMProvider:
    """
    A unified wrapper for LLM providers. It uses a configuration-driven approach
    to handle different models and output parsing formats.
    """
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = self._create_client()

    def _create_client(self) -> Any:
        # This factory handles any OpenAI-compatible API (vLLM, OpenRouter, etc.)
        api_key_env = self.config["api_config"].get("api_key_env")
        base_url = self.config["api_config"].get("base_url")
        api_key = os.environ.get(api_key_env) if api_key_env else None

        # Allow missing API key for localhost endpoints (common for local vLLM servers)
        if api_key_env and api_key is None:
            if isinstance(base_url, str) and ("localhost" in base_url or "127.0.0.1" in base_url):
                logging.warning(
                    "API key env '%s' not set, but base_url looks local (%s). Proceeding without a key.",
                    api_key_env,
                    base_url,
                )
                api_key = ""
            else:
                raise ValueError(f"API key environment variable '{api_key_env}' not set.")

        return ChatOpenAI(
            model=self.config["model_name"],
            base_url=base_url,
            api_key=api_key or "",  # Some OpenAI-compatible servers accept empty tokens for local use
            temperature=self.config["generation_config"].get("temperature", 0.7),
            max_tokens=self.config["generation_config"].get("max_tokens", 4096),
        )

    def _parse_with_think_tags(self, text: str) -> ModelOutput:
        """Parses text containing <think>...</think> tags."""
        if "<think>" in text:
            tag_word = "think"
        elif "<thinking>" in text:
            tag_word = "thinking"
        else:
            logging.warning(f"No thinking tags found in the response. Treating as direct response. The response is:\n--\n{text}\n--")
            return {"response": text, "thinking": ""}
        think_start_tag = f"<{tag_word}>"
        think_end_tag = f"</{tag_word}>"
        start_index = text.find(think_start_tag)
        end_index = text.find(think_end_tag)

        if start_index != -1 and end_index > start_index:
            thinking = text[start_index + len(think_start_tag):end_index].strip()
            response = text[end_index + len(think_end_tag):].strip()
            return {"response": response, "thinking": thinking}
        
        # If tags are not found, treat it as a direct response
        return {"response": text, "thinking": ""}

    def _parse_direct_response(self, text: str) -> ModelOutput:
        """Handles models that give a direct response with no thinking."""
        return {"response": text, "thinking": ""}
    
    def parse_response(self, full_response_text: str) -> ModelOutput:
        """
        NEW: Public method to parse a raw text response into a ModelOutput,
        extracting <think> tags based on the provider's configuration.
        """
        parsing_format = self.config.get("parsing_format", "direct_response")
        if parsing_format == "think_tags":
            return self._parse_with_think_tags(full_response_text)
        elif parsing_format == "direct_response":
            return self._parse_direct_response(full_response_text)
        else:
            logging.warning(f"Unsupported parsing format '{parsing_format}'. Defaulting to direct response.")
            return self._parse_direct_response(full_response_text)

    def invoke(self, messages: List[BaseMessage], thinking: bool = False) -> ModelOutput:
        """
        Invokes the model and parses the output based on the configured format.
        """
        try:
            response_obj = self.client.invoke(messages)
            full_response_text = response_obj.content
        except Exception as e:
            logging.error(f"An error occurred while calling the model server: {e}")
            return {"response": "Error: Could not connect to the model server.", "thinking": ""}
        
        # Parse the response based on the provider's configuration
        parsed_output = self.parse_response(full_response_text)

        # If thinking was not requested, clear the thinking trace from the final output
        if not thinking:
            parsed_output["thinking"] = ""
            
        return parsed_output


# --- Helper/Factory Function ---
def load_llm_provider(config_path: str, provider_name: str) -> LLMProvider:
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    provider_configs = full_config.get("providers", [])
    target_config = next((p for p in provider_configs if p["name"] == provider_name), None)
    
    if not target_config:
        raise ValueError(f"Provider '{provider_name}' not found in {config_path}")
        
    return LLMProvider(target_config)