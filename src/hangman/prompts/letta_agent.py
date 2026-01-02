"""
Letta agent memory block configurations.

Unlike Mem0/A-mem which use system prompts with memory injection,
Letta uses memory blocks that are always visible to the agent.

Letta's default system prompt already explains the memory tools comprehensively,
so we only customize the memory blocks to define the agent's role and task context.
"""

# Persona block: Defines the agent's identity and behavioral guidelines
PERSONA_BLOCK = """I am a helpful AI assistant that learns and evolves over time by managing my own memory to maintain consistency, continuity, and factual accuracy across turns."""

# Human block: Provides context about the user/task
HUMAN_BLOCK = """"""
