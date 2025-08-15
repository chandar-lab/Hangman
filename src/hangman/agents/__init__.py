"""
Initializes the agents package for the Hangman project.

This file provides a centralized factory function, `create_agent`, for instantiating
various agent implementations. It also exposes the base agent class and all
concrete agent classes for direct access if needed.

By using the `create_agent` factory, other parts of the application (like experiment
runners or benchmark scripts) can remain decoupled from the specific constructor
details of each agent.
"""

from typing import Optional, List

# --- Import Provider and Base Types for Type Hinting ---
from hangman.providers.llmprovider import LLMProvider
from hangman.agents.base_agent import BaseAgent, ModelOutput

# --- Import Concrete Agent Implementations ---
from hangman.agents.react_agent import ReActAgent
from hangman.agents.reactmem_agent import ReActMemAgent
from hangman.agents.private_cot_agent import PrivateCoTAgent
from hangman.agents.public_cot_agent import PublicCoTAgent
from hangman.agents.vanilla_llm_agent import VanillaLLMAgent
from hangman.agents.workflow_agent import WorkflowAgent
from hangman.tools import update_memory, get_search_tool, E2BCodeInterpreterTool

# --- Public API of the 'agents' package ---
__all__ = [
    "create_agent",
    "BaseAgent",
    "ModelOutput",
    "ReActAgent",
    "ReActMemAgent",
    "PrivateCoTAgent",
    "VanillaLLMAgent",
    "WorkflowAgent",
]

# --- Agent Factory ---

def create_agent(
    agent_name: str,
    main_llm_provider: LLMProvider,
    distillation_llm_provider: Optional[LLMProvider] = None,
    *,
    enable_websearch: bool = False,
    enable_code: bool = False,
    search_provider: str = "duckduckgo",
) -> BaseAgent:
    """
    Factory function to create an agent instance by its class name.

    Args:
        agent_name (str): The name of the agent class to instantiate (e.g., "ReActAgent").
        main_llm_provider (LLMProvider): The primary LLM provider for the agent's main actions.
        distillation_llm_provider (Optional[LLMProvider]): An optional LLM provider for agents
            that use a separate distillation/reasoning loop. Required for "ReaDis..." agents.

    Returns:
        BaseAgent: An instance of the requested agent.

    Raises:
        ValueError: If an unknown agent name is provided or if a required provider is missing.
    """
    # Agents that only require a single LLM provider
    if agent_name == "ReActAgent":
        tools: List = []
        if enable_websearch:
            tools.append(get_search_tool(provider=search_provider))
        if enable_code:
            tools.append(E2BCodeInterpreterTool().as_langchain_tool())
        return ReActAgent(main_llm_provider=main_llm_provider, tools=tools)
    elif agent_name == "ReActMemAgent":
        tools: List = [update_memory]
        if enable_websearch:
            tools.append(get_search_tool(provider=search_provider))
        if enable_code:
            tools.append(E2BCodeInterpreterTool().as_langchain_tool())
        return ReActMemAgent(main_llm_provider=main_llm_provider, tools=tools)
    elif agent_name in ["PrivateCoTAgent", "ProvatCoTAgent"]:
        # Backward compatibility alias: support "ProvatCoTAgent"
        return PrivateCoTAgent(main_llm_provider=main_llm_provider)

    # Agents that require a distillation LLM provider
    elif agent_name in ["ReaDisPatActAgent", "ReaDisOveActAgent", "ReaDisUpdActAgent"]:
        if not distillation_llm_provider:
            raise ValueError(
                f"Agent '{agent_name}' requires a 'distillation_llm_provider', but it was not provided."
            )
        
        if agent_name == "ReaDisPatActAgent":
            return ReaDisPatActAgent(main_llm_provider=main_llm_provider, distillation_llm_provider=distillation_llm_provider)
        elif agent_name == "ReaDisOveActAgent":
            return ReaDisOveActAgent(main_llm_provider=main_llm_provider, distillation_llm_provider=distillation_llm_provider)
        elif agent_name == "ReaDisUpdActAgent":
            return ReaDisUpdActAgent(main_llm_provider=main_llm_provider, distillation_llm_provider=distillation_llm_provider)

    # Handle unknown agent names
    else:
        known_agents = [name for name in __all__ if name.endswith("Agent")]
        raise ValueError(f"Unknown agent name: '{agent_name}'. Known agents are: {known_agents}")