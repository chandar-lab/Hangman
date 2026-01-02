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
from hangman.agents.reactmem_agent import ReActMemAgent
from hangman.agents.private_cot_agent import PrivateCoTAgent
from hangman.agents.vanilla_llm_agent import VanillaLLMAgent
from hangman.agents.workflow_agent import WorkflowAgent
from hangman.agents.mem0_agent import Mem0Agent
from hangman.agents.amem_agent import AMemAgent
from hangman.agents.lightmem_agent import LightMemAgent
from hangman.agents.memoryos_agent import MemoryOSAgent
from hangman.tools import update_memory

# --- Public API of the 'agents' package ---
__all__ = [
    "BaseAgent",
    "ModelOutput",
    "ReActMemAgent",
    "PrivateCoTAgent",
    "VanillaLLMAgent",
    "WorkflowAgent",
    "Mem0Agent",
    "AMemAgent",
    "LightMemAgent",
    "MemoryOSAgent",
]


def create_agent(
    agent_name: str,
    llm_provider: LLMProvider,
    distillation_llm_provider: Optional[LLMProvider] = None,
) -> BaseAgent:
    """
    Factory function to create an agent instance by its class name.

    Args:
        agent_name (str): The name of the agent class to instantiate (e.g., "ReActMemAgent").
        llm_provider (LLMProvider): The primary LLM provider for the agent's main actions.
        distillation_llm_provider (Optional[LLMProvider]): An optional LLM provider for agents
            that use a separate distillation/reasoning loop.

    Returns:
        BaseAgent: An instance of the requested agent.

    Raises:
        ValueError: If an unknown agent name is provided or if a required provider is missing.
    """
    if agent_name == "ReActMemAgent":
        tools: List = [update_memory]
        return ReActMemAgent(llm_provider=llm_provider, tools=tools)
    elif agent_name == "PrivateCoTAgent":
        return PrivateCoTAgent(llm_provider=llm_provider)
    elif agent_name == "VanillaLLMAgent":
        return VanillaLLMAgent(llm_provider=llm_provider)
    elif agent_name == "WorkflowAgent":
        return WorkflowAgent(llm_provider=llm_provider)
    elif agent_name == "Mem0Agent":
        return Mem0Agent(llm_provider=llm_provider)
    elif agent_name == "AMemAgent":
        return AMemAgent(llm_provider=llm_provider)
    elif agent_name == "LightMemAgent":
        return LightMemAgent(llm_provider=llm_provider)
    elif agent_name == "MemoryOSAgent":
        return MemoryOSAgent(llm_provider=llm_provider)
    else:
        known_agents = [name for name in __all__ if name.endswith("Agent")]
        raise ValueError(f"Unknown agent name: '{agent_name}'. Known agents are: {known_agents}")
