# Remember to run the Letta server before using this agent:
# letta server --host 127.0.0.1 --port 8283

import os
import yaml
from collections import deque
from typing import List, Any, Dict, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from hangman.agents.base_agent import BaseAgent, ModelOutput
from hangman.providers.llmprovider import LLMProvider, load_llm_provider
from hangman.prompts.letta_agent import PERSONA_BLOCK, HUMAN_BLOCK

try:
    from letta import create_client as Letta
except ImportError:
    try:
        from letta_client import Letta
    except ImportError:
        raise ImportError(
            "Letta client not found. Install with: pip install letta-client"
        )


# ------------------------------
# Agent State
# ------------------------------
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    thinking: str
    # Audit/debug fields:
    last_question: str
    last_system_prompt: str
    retrieved_memories: List[Dict[str, Any]]
    tool_calls_made: List[str]


# ------------------------------
# LettaAgent
# ------------------------------
class LettaAgent(BaseAgent):
    """
    Letta-based agent that uses a server-side memory system with:
      - Core memory blocks (human, persona) always in context
      - Recall memory (recent conversation history)
      - Archival memory (long-term vector storage triggered on context overflow)
      
    The agent wraps Letta's REST API in a LangGraph workflow to maintain
    compatibility with the BaseAgent interface.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        letta_config: Optional[Dict[str, Any]] = None,
        letta_config_path: Optional[str] = None,
        letta_base_url: str = "http://localhost:8283",
        session_id: str = "default",
        timeout: int = 1000,
        branch_id: Optional[str] = None,  # For SCT forking
        parent_session_id: Optional[str] = None,  # For SCT forking
        max_retries: int = 10,  # Number of retry attempts for API errors
    ):
        """
        Args:
            llm_provider: Pre-initialized provider (for compatibility; Letta uses its own)
            letta_config: (optional) dict containing Letta configuration
            letta_config_path: (optional) YAML path if you prefer a file
            letta_base_url: URL where Letta server is running
            session_id: Used to namespace agents per experiment
            timeout: HTTP timeout for Letta API calls
            branch_id: (optional) For SCT forking - creates a branch-specific session_id
            parent_session_id: (optional) For SCT forking - parent session to copy memories from
            max_retries: Maximum number of retry attempts for Letta API errors (default: 3)
        """
        # Load config from file if provided
        if letta_config is None and letta_config_path is not None:
            with open(letta_config_path, "r") as f:
                letta_config = yaml.safe_load(f)
        
        self.letta_config = letta_config or {}
        self.letta_config_path = letta_config_path
        self.letta_base_url = letta_base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Handle branching: if branch_id provided, create a unique session
        if branch_id is not None:
            base_session = parent_session_id if parent_session_id else session_id
            self.session_id = f"{base_session}__branch_{branch_id}"
            self.parent_session_id = base_session
            self.branch_id = branch_id
        else:
            self.session_id = session_id
            self.parent_session_id = None
            self.branch_id = None
        
        self._thread_id = f"letta_main__{self.session_id}"
        
        # Initialize Letta client
        self.letta_client = Letta(base_url=letta_base_url, timeout=timeout)
        
        # Extract config values with defaults
        llm_model = self.letta_config.get("llm", {}).get("model", "openai/gpt-oss-20b")
        llm_endpoint = self.letta_config.get("llm", {}).get("endpoint", "https://openrouter.ai/api/v1")
        embedding_model = self.letta_config.get("embedding", {}).get("model", "openai/text-embedding-3-large")
        embedding_endpoint = self.letta_config.get("embedding", {}).get("endpoint", "https://openrouter.ai/api/v1")
        context_window = self.letta_config.get("llm", {}).get("context_window", 4096)
        embedding_dim = self.letta_config.get("embedding", {}).get("dimension", 1536)
        
        # Create Letta agent with memory blocks from prompts
        # Letta's default system prompt is comprehensive, so we only customize memory blocks
        try:
            self.letta_agent = self.letta_client.agents.create(
                name=f"agent_{self.session_id}",
                agent_type="letta_v1_agent",
                tool_rules=[],
                llm_config={
                    "model": llm_model,
                    "model_endpoint_type": "openai",
                    "model_endpoint": llm_endpoint,
                    "context_window": context_window,
                },
                embedding_config={
                    "embedding_model": embedding_model,
                    "embedding_endpoint_type": "openai",
                    "embedding_endpoint": embedding_endpoint,
                    "embedding_dim": embedding_dim,
                },
                memory_blocks=[
                    {"label": "human", "value": HUMAN_BLOCK},
                    {"label": "persona", "value": PERSONA_BLOCK},
                ],
            )
            self.letta_agent_id = self.letta_agent.id
            print(f"✅ Letta agent created: {self.letta_agent_id}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create Letta agent. Is the Letta server running at {letta_base_url}?\n"
                f"Error: {e}"
            )
        
        # Store parent agent ID for cloning if this is a branch
        self._parent_letta_agent_id = None
        
        # Sliding window for maintaining conversation in LangGraph state
        self._window: deque = deque([], maxlen=20)
        
        # Parent init (sets self.llm_provider and builds workflow)
        super().__init__(llm_provider=llm_provider)
        
        # Reset state store
        self.reset()

    # ------------- LangGraph wiring -------------

    def _build_workflow(self) -> StateGraph:
        """Build a minimal StateGraph that wraps Letta API calls."""
        workflow = StateGraph(AgentState)
        workflow.add_node("letta_node", self._letta_node)
        workflow.set_entry_point("letta_node")
        return workflow.compile(checkpointer=MemorySaver())

    # ------------- Core logic -------------

    def _letta_node(self, state: AgentState) -> Dict[str, Any]:
        """
        One-step node that calls Letta API with retry logic:
        1) Extract latest user message
        2) Call Letta agent with message (with retries on failure)
        3) Parse response and extract reasoning
        4) Return updated state with AI message
        """
        incoming: List[BaseMessage] = state.get("messages", [])
        
        # Append new messages to sliding window
        for msg in incoming:
            if isinstance(msg, (HumanMessage, AIMessage)):
                self._window.append(msg)
        
        # Extract latest user message
        latest_user = ""
        for msg in reversed(incoming):
            if isinstance(msg, HumanMessage):
                latest_user = str(msg.content)
                break
        
        if not latest_user:
            # No user message to process
            return {
                "messages": state.get("messages", []),
                "thinking": "",
                "last_question": "",
                "retrieved_memories": [],
                "tool_calls_made": [],
            }
        
        # Call Letta API with retry logic
        response = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.letta_client.agents.messages.create(
                    agent_id=self.letta_agent_id,
                    messages=[{
                        "role": "user",
                        "content": [{"type": "text", "text": latest_user}]
                    }]
                )
                # Success - break out of retry loop
                break
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if this is a retryable error (tool call or validation errors)
                is_retryable = (
                    "No tool calls found in response" in error_str or
                    "validation error" in error_str.lower() or
                    "status_code: 400" in error_str
                )
                
                if is_retryable and attempt < self.max_retries - 1:
                    # Retry on tool call errors
                    print(f"⚠️ Letta API error (attempt {attempt + 1}/{self.max_retries}): {error_str[:100]}... Retrying...")
                    continue
                else:
                    # Non-retryable error or last attempt - propagate error
                    break
        
        # If all retries failed, return error message
        if response is None:
            error_msg = f"Error calling Letta API after {self.max_retries} attempts: {str(last_error)[:200]}"
            ai_msg = AIMessage(content=error_msg)
            return {
                "messages": state.get("messages", []) + [ai_msg],
                "thinking": f"API Error after {self.max_retries} retries: {last_error}",
                "last_question": latest_user,
                "retrieved_memories": [],
                "tool_calls_made": [],
            }
        
        # Extract response components
        final_response = ""
        reasoning = ""
        tool_calls = []
        retrieved_memories = []
        
        for msg in response.messages:
            msg_type = getattr(msg, 'message_type', None)
            
            if msg_type == "reasoning_message":
                reasoning = getattr(msg, 'reasoning', '')
            
            elif msg_type == "assistant_message":
                final_response = getattr(msg, 'content', '')
            
            elif msg_type == "tool_call_message":
                tool_call = getattr(msg, 'tool_call', None)
                if tool_call:
                    tool_name = getattr(tool_call, 'name', '')
                    tool_calls.append(tool_name)
            
            elif msg_type == "tool_return_message":
                # Track retrieval results
                tool_return = getattr(msg, 'tool_return', '')
                if tool_return and tool_return != "No results found.":
                    retrieved_memories.append({
                        "tool": getattr(msg, 'name', ''),
                        "result": tool_return[:500]  # Truncate long results
                    })
        
        # Create AI message
        ai_msg = AIMessage(content=final_response or "(no response)")
        
        return {
            "messages": state.get("messages", []) + [ai_msg],
            "thinking": reasoning,
            "last_question": latest_user,
            "retrieved_memories": retrieved_memories,
            "tool_calls_made": tool_calls,
        }

    # ------------- BaseAgent interface -------------

    def invoke(self, messages: List[BaseMessage]) -> ModelOutput:
        """One-turn run with a persistent thread_id."""
        cfg = {"configurable": {"thread_id": self._thread_id}}
        final_state = self.workflow.invoke({"messages": messages}, config=cfg)
        
        final_response = ""
        for m in reversed(final_state["messages"]):
            if isinstance(m, AIMessage):
                final_response = m.content
                break
        
        return {
            "response": final_response,
            "thinking": final_state.get("thinking", "")
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current LangGraph state."""
        snap = self.workflow.get_state({"configurable": {"thread_id": self._thread_id}})
        return snap.values if snap else {}

    def get_private_state(self) -> str:
        """
        Return Letta's server-side memory state.
        Queries core memory blocks and formats for display.
        """
        try:
            agent_state = self.letta_client.agents.retrieve(self.letta_agent_id)
            
            # Extract core memory blocks
            memory_sections = []
            
            if hasattr(agent_state, 'memory') and hasattr(agent_state.memory, 'blocks'):
                for block in agent_state.memory.blocks:
                    label = getattr(block, 'label', 'unknown')
                    value = getattr(block, 'value', '')
                    memory_sections.append(f"[Core Memory - {label.title()}]\n{value}")
            
            if not memory_sections:
                return "[Letta Memory]\n(empty or unavailable)"
            
            return "\n\n".join(memory_sections)
        
        except Exception as e:
            return f"<error retrieving Letta memory: {e}>"

    def reset(self) -> None:
        """Reset agent state by recreating Letta agent."""
        # Delete existing Letta agent
        try:
            self.letta_client.agents.delete(self.letta_agent_id)
        except Exception:
            pass  # Agent may not exist yet
        
        # Recreate with fresh memory (reuse config from __init__)
        llm_model = self.letta_config.get("llm", {}).get("model", "openai/gpt-oss-20b")
        llm_endpoint = self.letta_config.get("llm", {}).get("endpoint", "https://openrouter.ai/api/v1")
        embedding_model = self.letta_config.get("embedding", {}).get("model", "openai/text-embedding-3-large")
        embedding_endpoint = self.letta_config.get("embedding", {}).get("endpoint", "https://openrouter.ai/api/v1")
        context_window = self.letta_config.get("llm", {}).get("context_window", 4096)
        embedding_dim = self.letta_config.get("embedding", {}).get("dimension", 1536)
        
        try:
            self.letta_agent = self.letta_client.agents.create(
                name=f"agent_{self.session_id}",
                agent_type="letta_v1_agent",
                tool_rules=[],
                llm_config={
                    "model": llm_model,
                    "model_endpoint_type": "openai",
                    "model_endpoint": llm_endpoint,
                    "context_window": context_window,
                },
                embedding_config={
                    "embedding_model": embedding_model,
                    "embedding_endpoint_type": "openai",
                    "embedding_endpoint": embedding_endpoint,
                    "embedding_dim": embedding_dim,
                },
                memory_blocks=[
                    {"label": "human", "value": HUMAN_BLOCK},
                    {"label": "persona", "value": PERSONA_BLOCK},
                ],
            )
            self.letta_agent_id = self.letta_agent.id
        except Exception as e:
            print(f"Warning: Failed to recreate Letta agent: {e}")
        
        # Clear sliding window
        self._window.clear()
        
        # Reset LangGraph state
        empty: AgentState = AgentState(messages=[], thinking="")
        self.workflow.update_state(
            {"configurable": {"thread_id": self._thread_id}},
            empty
        )

    # ------------- SCT Forking Support -------------

    def clone_memories_from(self, parent_agent_id: str) -> None:
        """
        Clone full agent state (memory blocks + conversation history) from parent 
        Letta agent using export/import.
        
        This approach:
        - Exports parent agent's complete state (memory + messages)
        - Modifies agent ID and name for the branch
        - Imports as a new agent, preserving full conversation history
        - NO message replay = NO LLM contamination
        
        Used for SCT forking to ensure each branch starts with identical pre-fork state.
        
        Args:
            parent_agent_id: The Letta agent ID to copy state from
        """
        import json
        import io
        import copy
        import random
        import time
        
        try:
            # 1. Export parent agent (includes memory + full conversation history)
            print(f"📤 Exporting parent agent {parent_agent_id[:12]}...")
            export_data = self.letta_client.agents.export_file(agent_id=parent_agent_id)
            
            # 2. Deep copy and modify for branch agent
            branch_export = copy.deepcopy(export_data)
            
            if 'agents' in branch_export and len(branch_export['agents']) > 0:
                branch_agent_data = branch_export['agents'][0]
                
                # Save old agent ID for message sender updates
                old_agent_id = branch_agent_data['id']
                
                # Generate new ID in Letta's format: agent-{integer}
                new_agent_id = f"agent-{int(time.time() * 1000) + random.randint(0, 9999)}"
                branch_agent_data['id'] = new_agent_id
                branch_agent_data['name'] = f"branch_{self.session_id}"
                
                # Update message sender_ids to reference the new agent
                if 'messages' in branch_agent_data:
                    for msg in branch_agent_data['messages']:
                        if msg.get('sender_id') == old_agent_id:
                            msg['sender_id'] = new_agent_id
                
                print(f"📝 Modified export: {old_agent_id[:12]} → {new_agent_id[:12]}")
                
                # 3. Delete current branch agent (created in __init__)
                try:
                    self.letta_client.agents.delete(agent_id=self.letta_agent_id)
                    print(f"🗑️ Deleted placeholder agent {self.letta_agent_id[:12]}")
                except Exception as e:
                    print(f"⚠️ Could not delete placeholder agent: {e}")
                
                # 4. Import as new branch agent
                print(f"📥 Importing branch agent...")
                json_str = json.dumps(branch_export)
                json_bytes = json_str.encode('utf-8')
                file_obj = io.BytesIO(json_bytes)
                
                import_result = self.letta_client.agents.import_file(file=file_obj)
                
                # 5. Update this agent's ID to the imported one
                if hasattr(import_result, 'agent_ids') and import_result.agent_ids:
                    self.letta_agent_id = import_result.agent_ids[0]
                    print(f"✅ Branch agent created: {self.letta_agent_id[:12]}")
                    
                    # Verify conversation history was preserved
                    imported_messages = self.letta_client.agents.messages.list(agent_id=self.letta_agent_id)
                    print(f"✅ Cloned {len(imported_messages)} messages from parent")
                else:
                    raise RuntimeError("Import did not return agent_ids")
            
            else:
                raise RuntimeError("Export data missing 'agents' field")
        
        except Exception as e:
            # Critical error - SCT needs proper branching
            raise RuntimeError(
                f"Failed to clone agent state from {parent_agent_id}: {e}\n"
                f"SCT branching requires export/import to work correctly."
            )
    
    def get_session_config(self) -> Dict[str, Any]:
        """
        Return configuration needed to create a branch agent.
        Used by engine_sct.py to instantiate branch agents with the same settings.
        """
        return {
            "llm_provider": self.llm_provider,
            "letta_config": self.letta_config,
            "letta_config_path": self.letta_config_path,
            "letta_base_url": self.letta_base_url,
            "session_id": self.parent_session_id or self.session_id,  # Use base session
            "timeout": self.timeout,
            "max_retries": self.max_retries,  # Include retry config
            "parent_letta_agent_id": self.letta_agent_id,  # For cloning
        }
    
    def get_sliding_window_state(self) -> List[BaseMessage]:
        """
        Return the current sliding window messages.
        Used by engine_sct.py to seed branch agents with pre-fork conversation state.
        """
        return list(self._window)

    def __del__(self):
        """Cleanup: delete Letta agent when Python object is destroyed."""
        try:
            if hasattr(self, 'letta_client') and hasattr(self, 'letta_agent_id'):
                self.letta_client.agents.delete(self.letta_agent_id)
        except Exception:
            pass  # Ignore cleanup errors


# --- Runnable CLI for Direct Testing ---
if __name__ == "__main__":
    CONFIG_PATH = "config/config.yaml"  # LLM provider config
    LETTA_CONFIG_PATH = "config/letta_config_gptoss_20b.yaml"  # Letta-specific config
    
    print("Is config file readable:", os.access(CONFIG_PATH, os.R_OK))
    print("Is Letta config file readable:", os.access(LETTA_CONFIG_PATH, os.R_OK))
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    # Load your OpenRouter-backed LLMProvider (for compatibility)
    try:
        main_llm = load_llm_provider(CONFIG_PATH, provider_name="gpt_oss_20b_openrouter")
        print("✅ LLM Provider loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load LLM Provider: {e}")
        raise SystemExit(1)
    
    # Initialize the LettaAgent
    try:
        agent = LettaAgent(
            llm_provider=main_llm,
            letta_config_path=LETTA_CONFIG_PATH,
            letta_base_url="http://localhost:8283",
            session_id="letta_session_1",
            timeout=1000,
        )
        print("🤖 LettaAgent is ready. Type 'quit', 'exit', or 'q' to end.")
    except Exception as e:
        print(f"❌ Failed to initialize LettaAgent: {e}")
        print("Make sure the Letta server is running:")
        print("  letta server --host 127.0.0.1 --port 8283")
        raise SystemExit(1)
    
    # Interactive loop
    messages = []
    while True:
        try:
            user_input = input("\nUser > ")
        except (EOFError, KeyboardInterrupt):
            print("\nEnding session.")
            break
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Ending session.")
            break
        
        messages.append(HumanMessage(content=user_input))
        
        output = agent.invoke(messages)
        
        messages.append(AIMessage(content=output["response"]))
        
        print("\n---ANSWER---")
        print(f"AI: {output['response']}")
        
        # Print thinking trace if available
        if "thinking" in output and output["thinking"]:
            print("\n---THINKING TRACE---")
            print(output["thinking"])
        
        print("\n---LETTA MEMORY STATE---")
        print(agent.get_private_state())
        
        # Show tool usage from last state
        state = agent.get_state()
        if state.get("tool_calls_made"):
            print("\n---TOOLS USED---")
            print(", ".join(state["tool_calls_made"]))
        
        print("\n" + "=" * 50)
    
    # Cleanup
    print("\n🗑️ Cleaning up Letta agent...")
    try:
        agent.letta_client.agents.delete(agent.letta_agent_id)
        print("✅ Agent deleted successfully.")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
